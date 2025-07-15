from ..utils import Strategy
from dataclasses import dataclass, field
from ..cost_model.profile_data_parser import ProfileDataParser, ProfileDataParserArguments
import math
from typing import List

@dataclass
class SearchEngineArguments:
    search_granularity: str = field(default="coarse-grained", metadata={"help": "The granularity of the search space."})
    world_size: int = field(default=8, metadata={"help": "The number of processes to use for distributed training."})

    min_bsz: int = field(default=64, metadata={"help": "The minimum batch size."})
    max_bsz: int = field(default=64, metadata={"help": "The maximum batch size."})
    bsz_step: int = field(default=1, metadata={"help": "The step size for batch size."})
    
    max_tp_size: int = field(default=8, metadata={"help": "The maximum tensor parallel size."})
    max_pp_size: int = field(default=8, metadata={"help": "The maximum pipeline parallel size."})
    
    mixed_precision_type: str = field(default="bp16", metadata={"help": "The mixed precision type to use."})
    memory_upper_limit: int = field(default=24, metadata={"help": "The upper limit of memory usage in GB"})
    
    # Layerwise recompute search parameters
    enable_fine_grained_recompute_search: bool = field(default=True, metadata={"help": "Enable layerwise recompute search"})
    total_layers: int = field(default=16, metadata={"help": "Total number of layers in the model"})
    enable_layerwise_recompute: bool = field(default=True, metadata={"help": "Enable layerwise recompute analysis and output"})
    enable_memory_efficiency_mode: bool = field(default=True, metadata={"help": "Enable memory efficiency optimization: prefer less recompute when memory is sufficient"})
    
    def initialize(self, args_dict):
        self.search_granularity = args_dict.get("--search_granularity", self.search_granularity)
        self.world_size = int(args_dict.get("--world_size", self.world_size))
        self.min_bsz = int(args_dict.get("--min_bsz", self.min_bsz))
        self.max_bsz = int(args_dict.get("--max_bsz", self.max_bsz))
        self.bsz_step = int(args_dict.get("--bsz_step", self.bsz_step))
        self.max_tp_size = int(args_dict.get("--max_tp_size", self.max_tp_size))
        self.max_pp_size = int(args_dict.get("--max_pp_size", self.max_pp_size))
        self.mixed_precision_type = args_dict.get("--mixed_precision_type", self.mixed_precision_type)
        self.memory_upper_limit = int(args_dict.get("--memory_upper_limit", self.memory_upper_limit))
        
        # Layerwise recompute parameters
        self.enable_fine_grained_recompute_search = bool(args_dict.get("--enable_fine_grained_recompute_search", self.enable_fine_grained_recompute_search))
        self.total_layers = int(args_dict.get("--total_layers", self.total_layers))
        self.enable_layerwise_recompute = bool(args_dict.get("--enable_layerwise_recompute", self.enable_layerwise_recompute))
        self.enable_memory_efficiency_mode = bool(args_dict.get("--enable_memory_efficiency_mode", self.enable_memory_efficiency_mode))
        
class SearchEngine:
    def __init__(self, args_dict):
        self.args = SearchEngineArguments()
        self.args.initialize(args_dict)
        
        parser_data_args = ProfileDataParserArguments()
        parser_data_args.initialize(args_dict)
        self.parser = ProfileDataParser(parser_data_args)
        
        self.generate_strategies()
        self.set_searching_bsz()

    def generate_strategies(self):
        print('Searching parallism configs....\n')
        args = self.args
        self.strategy_set = []  # 修复：初始化self.strategy_set而不是局部变量
        
        # Generate parallelism configurations with priority ordering
        # Priority: lower pp_size (better performance), balanced tp/dp
        parallelism_configs = []
        
        for pp_size in range(1, min(args.max_pp_size + 1, args.world_size + 1)):
            for tp_size in range(1, min(args.max_tp_size + 1, args.world_size + 1)):
                if tp_size * pp_size > args.world_size:
                    continue
                dp_size = args.world_size // (tp_size * pp_size)
                if tp_size * pp_size * dp_size == args.world_size:
                    # Calculate priority score: prefer smaller pp_size, balanced tp/dp
                    priority = pp_size * 100 + abs(tp_size - dp_size)  # Lower is better
                    parallelism_configs.append((priority, pp_size, tp_size, dp_size))
        
        # Sort by priority (lower pp_size first, then balanced configurations)
        parallelism_configs.sort(key=lambda x: x[0])
        
        print(f"Parallelism configurations (ordered by priority):")
        for i, (priority, pp_size, tp_size, dp_size) in enumerate(parallelism_configs[:10]):
            print(f"   {i+1}. pp{pp_size}_tp{tp_size}_dp{dp_size} (priority: {priority})")
        if len(parallelism_configs) > 10:
            print(f"   ... and {len(parallelism_configs) - 10} more")
        
        # For high-priority configurations, generate more comprehensive mixed strategies
        high_priority_threshold = 3  # First 3 configurations get comprehensive search
        
        for config_idx, (priority, pp_size, tp_size, dp_size) in enumerate(parallelism_configs):
            is_high_priority = config_idx < high_priority_threshold
            if is_high_priority:
                print(f"\nGenerating comprehensive strategies for high-priority config: pp{pp_size}_tp{tp_size}_dp{dp_size}")
            
            sharding_stage_set = [0]
            if dp_size > 1:
                sharding_stage_set = [0, 1, 2, 3]
                
            # when in static mode, RuntimeError: Operation((%0) = "pd_op.embedding_grad" is not support sharded by shard_tensor op in pir mode happend.
            for recompute in [0, 1]:
                for sharding_stage in sharding_stage_set:
                    if recompute == 0:
                        # No recompute case - this should be fastest when memory allows
                        if args.enable_layerwise_recompute:
                            layerwise_recompute = [0] * args.total_layers  # No layers recompute
                        else:
                            layerwise_recompute = []
                        strategy = Strategy(pp_size=pp_size, tp_size=tp_size, dp_size=dp_size, sharding_stage=sharding_stage, recompute=recompute, layerwise_recompute=layerwise_recompute)
                        self.strategy_set.append(strategy)
                        if is_high_priority:
                            print(f"Added no-recompute strategy: pp{pp_size}_tp{tp_size}_dp{dp_size}_stage{sharding_stage}")
                    else:
                        # Recompute enabled: explore fine-grained configurations
                        if args.enable_fine_grained_recompute_search:
                            self._generate_fine_grained_recompute_strategies(pp_size, tp_size, dp_size, sharding_stage, is_high_priority)
                        else:
                            # Default recompute strategy
                            strategy = Strategy(pp_size=pp_size, tp_size=tp_size, dp_size=dp_size, sharding_stage=sharding_stage, recompute=recompute)
                            self.strategy_set.append(strategy)
                            
        print(f'SearchEngine strategy_set size: {len(self.strategy_set)}')
        print('Sample strategies:')
        for i, strategy in enumerate(self.strategy_set[:10]):
            print(f'  {i+1}: {strategy.serialize()}')
        if len(self.strategy_set) > 10:
            print(f'  ... and {len(self.strategy_set) - 10} more')
        print()
    
    def _generate_fine_grained_recompute_strategies(self, pp_size, tp_size, dp_size, sharding_stage, is_high_priority=False):
        """Generate layerwise recompute strategies based on layer count"""
        args = self.args
        
        # For high-priority configurations, generate more comprehensive strategies
        if is_high_priority:
            print(f"Generating comprehensive layerwise strategies for pp{pp_size}_tp{tp_size}_dp{dp_size}")
        
        # Generate layerwise strategies based on recompute layer count
        if args.enable_layerwise_recompute:
            print(f"Using optimal layerwise strategies for layer count based search")
            before_count = len(self.strategy_set)
            self._generate_optimal_layerwise_recompute_strategies(pp_size, tp_size, dp_size, sharding_stage)
            after_count = len(self.strategy_set)
            print(f"Added {after_count - before_count} recompute strategies (1 to {args.total_layers} layers)")
        else:
            # Fallback: Strategy with full recompute
            strategy = Strategy(
                pp_size=pp_size, tp_size=tp_size, dp_size=dp_size, 
                sharding_stage=sharding_stage, recompute=1,
            recompute_granularity="full",
                layerwise_recompute=[]
            )
            self.strategy_set.append(strategy)
            
            # Pipeline recompute interval strategies (only for pp_size > 1)
            if pp_size > 1:
                for pp_interval in [2, 3, 4]:
                    if pp_interval <= pp_size:
                        # Generate default layerwise pattern for pipeline recompute
                        if args.enable_layerwise_recompute:
                            layerwise_recompute = [1] * args.total_layers  # All layers recompute by default
                        else:
                            layerwise_recompute = []
                        
                        strategy = Strategy(
                            pp_size=pp_size, tp_size=tp_size, dp_size=dp_size,
                            sharding_stage=sharding_stage, recompute=1,
                        recompute_granularity="full",
                            pp_recompute_interval=pp_interval,
                            layerwise_recompute=layerwise_recompute
                        )
                        self.strategy_set.append(strategy)
    

    
    def _generate_optimal_layerwise_recompute_strategies(self, pp_size, tp_size, dp_size, sharding_stage):
        """
        Generate layerwise recompute strategies based on recompute layer count.
        Uses simple pattern: first N layers recompute, since which specific layers 
        recompute doesn't significantly impact performance.
        Note: Skip recompute_count=0 since it's already handled by main loop with recompute=0.
        """
        args = self.args
        total_layers = args.total_layers
        
        print(f"Generating layerwise strategies for {total_layers} layers (first-N-layers pattern)")
        
        # Generate strategies with different numbers of recompute layers
        # Start from 1 (not 0, since recompute=0 is handled in main loop) and go up to full recompute
        for recompute_count in range(1, total_layers + 1):
            layerwise_recompute = [1 if i < recompute_count else 0 for i in range(total_layers)]
            
            # Create strategy with recompute=1 since we have at least 1 layer to recompute
            strategy = Strategy(
                pp_size=pp_size, tp_size=tp_size, dp_size=dp_size,
                sharding_stage=sharding_stage, recompute=1,
                recompute_granularity="full",
                layerwise_recompute=layerwise_recompute
            )
            self.strategy_set.append(strategy)
            
           
        print(f"Generated {total_layers} layerwise strategies (1 to {total_layers} recompute layers)")
    
    def _get_recompute_ratio(self, strategy):
        """Calculate the recompute ratio for a strategy (0.0 = no recompute, 1.0 = full recompute)"""
        if not strategy.recompute:
            return 0.0
            
        # If layerwise_recompute is available, use it for precise calculation
        if hasattr(strategy, 'layerwise_recompute') and strategy.layerwise_recompute:
            total_layers = len(strategy.layerwise_recompute)
            recompute_layers = sum(strategy.layerwise_recompute)
            return recompute_layers / total_layers if total_layers > 0 else 1.0
        
        # If no_recompute_layers is available, calculate from that
        if hasattr(strategy, 'no_recompute_layers') and strategy.no_recompute_layers:
            total_layers = self.args.total_layers
            recompute_layers = total_layers - len(strategy.no_recompute_layers)
            return recompute_layers / total_layers if total_layers > 0 else 1.0
        
        # Default: full recompute
        return 1.0
    
    def _analyze_skip_pattern(self, no_recompute_layers, total_layers):
        """Analyze the pattern of skipped layers"""
        if not no_recompute_layers:
            return "none"
        
        layers = sorted(no_recompute_layers)
        
        # Check if it's consecutive from start
        if layers == list(range(len(layers))):
            return f"first_{len(layers)}_layers"
        
        # Check if it's consecutive from end
        if layers == list(range(total_layers - len(layers), total_layers)):
            return f"last_{len(layers)}_layers"
        
        # Check if it's in the middle
        if len(layers) > 1 and layers == list(range(layers[0], layers[0] + len(layers))):
            start_pos = layers[0] / total_layers
            if 0.2 < start_pos < 0.8:  # Roughly in the middle
                return f"middle_{len(layers)}_layers_from_{layers[0]}"
        
        # Check if it's a regular pattern (every n-th layer)
        if len(layers) > 2:
            diffs = [layers[i+1] - layers[i] for i in range(len(layers)-1)]
            if len(set(diffs)) == 1:  # All differences are the same
                return f"every_{diffs[0]}_layers_{len(layers)}_total"
        
        # Custom pattern
        return f"custom_pattern_{layers}"
    
    def _analyze_recompute_strategy(self, strategy, total_layers):
        """Analyze and explain the recompute strategy"""
        analysis = {
            'enabled': True,
            'granularity': getattr(strategy, 'recompute_granularity', 'full'),
            'total_layers': total_layers
        }
        
        granularity_info = {
            'full': 'Recompute all forward activations during backward pass',
            'core_attn': 'Recompute only attention core operations (~50% compute overhead)',
            'full_attn': 'Recompute entire attention modules (~70% compute overhead)'
        }
        
        analysis['granularity_description'] = granularity_info.get(
            analysis['granularity'], 
            'Custom recompute granularity'
        )
        
        # Analyze no-recompute layers
        if hasattr(strategy, 'no_recompute_layers') and strategy.no_recompute_layers:
            skip_layers = strategy.no_recompute_layers
            analysis['skip_layers'] = {
                'layers': skip_layers,
                'count': len(skip_layers),
                'percentage': round(len(skip_layers) / total_layers * 100, 1),
                'pattern': self._analyze_skip_pattern(skip_layers, total_layers)
            }
            
            # Calculate memory savings estimate
            recompute_ratio = (total_layers - len(skip_layers)) / total_layers
            if analysis['granularity'] == 'core_attn':
                overhead_reduction = recompute_ratio * 0.5
            elif analysis['granularity'] == 'full_attn':
                overhead_reduction = recompute_ratio * 0.7
            else:  # full
                overhead_reduction = recompute_ratio * 1.0
                
            analysis['estimated_compute_overhead'] = f"{overhead_reduction:.1%}"
            analysis['strategy_explanation'] = f"Skip {len(skip_layers)} layers to reduce recompute overhead while saving memory on remaining {total_layers - len(skip_layers)} layers"
        else:
            analysis['skip_layers'] = None
            if analysis['granularity'] == 'core_attn':
                analysis['estimated_compute_overhead'] = "50%"
            elif analysis['granularity'] == 'full_attn':
                analysis['estimated_compute_overhead'] = "70%"
            else:
                analysis['estimated_compute_overhead'] = "100%"
            analysis['strategy_explanation'] = f"Full {analysis['granularity']} recompute on all {total_layers} layers"
        
        # Pipeline recompute interval
        if hasattr(strategy, 'pp_recompute_interval') and strategy.pp_recompute_interval:
            analysis['pipeline_interval'] = strategy.pp_recompute_interval
            analysis['pipeline_explanation'] = f"Recompute every {strategy.pp_recompute_interval} pipeline stages"
        
        # Layerwise recompute pattern
        if hasattr(strategy, 'layerwise_recompute') and strategy.layerwise_recompute:
            layerwise_pattern = strategy.layerwise_recompute
            analysis['layerwise_recompute'] = {
                'pattern': layerwise_pattern,
                'total_layers': len(layerwise_pattern),
                'recompute_layers': [i for i, val in enumerate(layerwise_pattern) if val == 1],
                'no_recompute_layers': [i for i, val in enumerate(layerwise_pattern) if val == 0],
                'recompute_count': sum(layerwise_pattern),
                'no_recompute_count': len(layerwise_pattern) - sum(layerwise_pattern),
                'recompute_ratio': sum(layerwise_pattern) / len(layerwise_pattern)
            }
            
            # Update strategy explanation for layerwise
            recompute_ratio = analysis['layerwise_recompute']['recompute_ratio']
            analysis['strategy_explanation'] = f"Layerwise recompute pattern {layerwise_pattern}: " \
                                              f"{analysis['layerwise_recompute']['recompute_count']} layers recompute, " \
                                              f"{analysis['layerwise_recompute']['no_recompute_count']} layers skip"
            
            # Adjust compute overhead based on actual recompute ratio
            if analysis['granularity'] == 'core_attn':
                base_overhead = 0.5
            elif analysis['granularity'] == 'full_attn':
                base_overhead = 0.7
            else:
                base_overhead = 1.0
            
            effective_overhead = base_overhead * recompute_ratio
            analysis['estimated_compute_overhead'] = f"{effective_overhead:.1%}"
        
        return analysis
    
    def set_searching_bsz(self):
        args = self.args
        min_bsz, max_bsz, bsz_step = args.min_bsz, args.max_bsz, args.bsz_step
        min_bsz = max(min_bsz, bsz_step)
        min_bsz = min_bsz // bsz_step * bsz_step
        max_bsz = int(math.ceil(max_bsz / bsz_step) * bsz_step) if max_bsz % bsz_step != 0 else max_bsz + bsz_step
        self.BSZs = list(range(min_bsz, max_bsz, bsz_step))
        
        # change the min_bsz and max_bsz
        args.min_bsz = min_bsz
        args.max_bsz = max_bsz
        
        print('-----', '[Searching Batch Sizes Info]', 'Min bsz:', args.min_bsz, 'Max bsz:', args.max_bsz, 'bsz_step:', args.bsz_step, '-----')
        print('Searching Batch Sizes:', self.BSZs)
      
    def parallelism_optimization(self):
        args = self.args
        
        if args.search_granularity == 'coarse-grained':
            optimal_solution, max_throughput, optimal_history = {}, -1, []
            results = dict()
            for bsz in self.BSZs:
                results[bsz] = dict()
                accumulation_steps_list = range(1, bsz + 1)
                for accumulation_steps in accumulation_steps_list:
                    results[bsz][accumulation_steps] = dict()
                    if bsz % accumulation_steps != 0:
                        continue
                    for strategy in self.strategy_set:
                        results[bsz][accumulation_steps][strategy.serialize()] = dict()
                        if bsz // accumulation_steps < strategy.dp_size:
                            continue
                        memory_cost = self.parser.get_memory_cost_for_specific_strategy(strategy, bsz, args.mixed_precision_type, accumulation_steps)
                        time_cost = self.parser.get_time_cost_for_specific_strategy(strategy, bsz, args.mixed_precision_type, accumulation_steps)
                        results[bsz][accumulation_steps][strategy.serialize()]['memory_cost'] = memory_cost
                        results[bsz][accumulation_steps][strategy.serialize()]['time_cost'] = time_cost
                        results[bsz][accumulation_steps][strategy.serialize()]['throughput'] = bsz / time_cost if time_cost > 0 else 0
                        # Add safety margin: memory should be at most 90% of limit to avoid edge cases
                        memory_limit_with_safety = args.memory_upper_limit * 1024 * 0.90  # 90% of limit for safety
                        results[bsz][accumulation_steps][strategy.serialize()]['OOM'] = memory_cost[0] > memory_limit_with_safety # memory_cost[0] means the first stage memory cost
                        
                        current_throughput = results[bsz][accumulation_steps][strategy.serialize()]['throughput']
                        is_oom = results[bsz][accumulation_steps][strategy.serialize()]['OOM']
                        
                        # 🔍 Debug: Track recompute strategies performance
                        if hasattr(strategy, 'layerwise_recompute') and strategy.layerwise_recompute:
                            layerwise_pattern = strategy.layerwise_recompute
                            recompute_count = sum(layerwise_pattern)
                            total_layers = len(layerwise_pattern)
                            memory_gb = memory_cost[0] / 1024
                            # Show ALL mixed patterns, not just non-OOM ones, and include parallelism config
                            if recompute_count not in [0, total_layers]:  # Only mixed patterns
                                parallelism_config = f"pp{strategy.pp_size}_tp{strategy.tp_size}_dp{strategy.dp_size}"
                                oom_status = "OOM" if is_oom else "OK"
                                print(f'   🎯 Mixed recompute strategy: {layerwise_pattern} ({recompute_count}/{total_layers} layers), {parallelism_config}, throughput={current_throughput:.1f}, memory={memory_gb:.1f}GB, status={oom_status}')
                        
                        # Optimized strategy selection logic with minimal recompute preference
                        should_update = False
                        if not is_oom:
                            if optimal_solution == {}:  # First valid solution
                                should_update = True
                            else:
                                # Calculate recompute counts for both strategies
                                def get_recompute_count(strat):
                                    if not strat.recompute:
                                        return 0
                                    if hasattr(strat, 'layerwise_recompute') and strat.layerwise_recompute:
                                        return sum(strat.layerwise_recompute)
                                    return args.total_layers  # Full recompute as fallback
                                
                                current_recompute_count = get_recompute_count(strategy)
                                optimal_recompute_count = get_recompute_count(optimal_solution['strategy'])
                                
                                # Calculate memory utilization ratio (with safety margin)
                                memory_utilization = memory_cost[0] / (args.memory_upper_limit * 1024 * 0.90)  # Use 90% limit for safety
                                optimal_memory_cost = optimal_solution['memory_cost'][0] 
                                current_memory_cost = memory_cost[0]
                                
                                # Throughput improvement threshold
                                throughput_threshold = 0.02  # 2% threshold for fairness
                                throughput_ratio = current_throughput / max_throughput if max_throughput > 0 else 0
                                
                                # Primary rule: Prefer strategies with fewer recompute layers
                                if current_recompute_count < optimal_recompute_count:
                                    # Current strategy uses less recompute - prefer it unless performance is significantly worse
                                    # Allow more tolerance when memory is not critically constrained
                                    if memory_utilization < 0.9:
                                        tolerance = 0.05  # 5% tolerance when memory is abundant
                                    elif memory_utilization < 0.95:
                                        tolerance = 0.03  # 3% tolerance when memory is comfortable
                                    else:
                                        tolerance = 0.01  # 1% tolerance when memory is tight
                                    
                                    if throughput_ratio >= (1 - tolerance):
                                        should_update = True
                                        print(f"   🎯 Choosing minimal recompute strategy: {current_recompute_count} vs {optimal_recompute_count} layers, throughput ratio: {throughput_ratio:.3f}, memory: {memory_utilization:.1%}")
                                
                                elif current_recompute_count > optimal_recompute_count:
                                    # Current strategy uses more recompute - only choose if significantly better performance
                                    if throughput_ratio > (1 + throughput_threshold * 2):
                                        should_update = True
                                        print(f"   ⚡ Choosing higher-recompute strategy due to significant performance gain: {current_recompute_count} vs {optimal_recompute_count} layers, throughput ratio: {throughput_ratio:.3f}")
                                
                                else:
                                    # Same recompute count: choose higher throughput
                                    if current_throughput > max_throughput:
                                        should_update = True
                                        print(f"   📈 Choosing better throughput with same recompute count ({current_recompute_count} layers): {throughput_ratio:.3f}")
                                
                                # Special case: no recompute vs any recompute - strongly prefer no recompute
                                if current_recompute_count == 0 and optimal_recompute_count > 0:
                                    if throughput_ratio >= 0.90:  # Only need 90% performance to prefer no recompute
                                        should_update = True
                                        print(f"   ✨ Strongly preferring no-recompute strategy (throughput ratio: {throughput_ratio:.3f}, memory: {memory_utilization:.1%})")
                        
                        if should_update:
                            max_throughput = current_throughput
                            optimal_solution = {
                                'bsz': bsz,
                                'accumulation_steps': accumulation_steps,
                                'strategy': strategy,
                                'memory_cost': memory_cost,
                                'time_cost': time_cost,
                                'throughput': max_throughput
                            }
                            optimal_history.append(optimal_solution)
                            
                            # 🔍 Debug: Show strategy update details
                            layerwise_info = getattr(strategy, 'layerwise_recompute', 'N/A')
                            recompute_ratio = sum(layerwise_info) / len(layerwise_info) if isinstance(layerwise_info, list) else 'N/A'
                            print(f'   🔄 UPDATED optimal: {strategy.serialize()}, layerwise={layerwise_info}, recompute_ratio={recompute_ratio}, throughput={current_throughput:.2f}')
                        print(f'Batch Size: {bsz}, Accumulation Steps: {accumulation_steps}, Strategy: {strategy.serialize()}, Memory Cost: {memory_cost} MB, Time Cost: {time_cost} s, Throughput: {results[bsz][accumulation_steps][strategy.serialize()]["throughput"]} Sample/s, OOM: {results[bsz][accumulation_steps][strategy.serialize()]["OOM"]}')
            print('-----', '[Optimal Solution History]', '-----')
            for history in optimal_history:
                print(f'Batch Size: {history["bsz"]}, Accumulation Steps: {history["accumulation_steps"]}, Strategy: {history["strategy"].serialize()}, Memory Cost: {history["memory_cost"]} MB, Time Cost: {history["time_cost"]} s, Throughput: {history["throughput"]} Sample/s')
            print('-----', '[Optimal Solution]', '-----')
            print('Optimal Solution:', optimal_solution)
            
            import os 
            current_dir = os.getcwd()
            optimal_solution_path = os.path.join(current_dir, './configs/optimal_solution.json')
            with open(optimal_solution_path, 'w') as f:
                import json
                if optimal_solution:  # Check if optimal_solution is not empty
                    strategy = optimal_solution['strategy']
                    
                    # Extract detailed strategy information
                    strategy_details = {
                        'pp_size': strategy.pp_size,
                        'tp_size': strategy.tp_size,
                        'dp_size': strategy.dp_size,
                        'sharding_stage': strategy.sharding_stage,
                        'recompute': strategy.recompute,
                        'strategy_name': strategy.serialize()
                    }
                    
                    # Add fine-grained recompute information if available
                    if hasattr(strategy, 'recompute_granularity') and strategy.recompute:
                        strategy_details['recompute_granularity'] = strategy.recompute_granularity
                    
                    if hasattr(strategy, 'no_recompute_layers') and strategy.no_recompute_layers:
                        strategy_details['no_recompute_layers'] = strategy.no_recompute_layers
                        strategy_details['num_no_recompute_layers'] = len(strategy.no_recompute_layers)
                        strategy_details['no_recompute_pattern'] = self._analyze_skip_pattern(strategy.no_recompute_layers, args.total_layers)
                    
                    if hasattr(strategy, 'pp_recompute_interval') and strategy.pp_recompute_interval:
                        strategy_details['pp_recompute_interval'] = strategy.pp_recompute_interval
                    
                    # Add layerwise recompute information
                    if hasattr(strategy, 'layerwise_recompute'):
                        if strategy.recompute and strategy.layerwise_recompute:
                            strategy_details['layerwise_recompute'] = strategy.layerwise_recompute
                            strategy_details['layerwise_recompute_layers'] = [i for i, val in enumerate(strategy.layerwise_recompute) if val == 1]
                            strategy_details['layerwise_no_recompute_layers'] = [i for i, val in enumerate(strategy.layerwise_recompute) if val == 0]
                            strategy_details['recompute_layers_count'] = sum(strategy.layerwise_recompute)
                            strategy_details['no_recompute_layers_count'] = len(strategy.layerwise_recompute) - sum(strategy.layerwise_recompute)
                            strategy_details['recompute_ratio'] = sum(strategy.layerwise_recompute) / len(strategy.layerwise_recompute) if len(strategy.layerwise_recompute) > 0 else 0.0
                        else:
                            # No recompute or empty layerwise pattern
                            strategy_details['layerwise_recompute'] = [0] * args.total_layers if not strategy.recompute else []
                            strategy_details['layerwise_recompute_layers'] = []
                            strategy_details['layerwise_no_recompute_layers'] = list(range(args.total_layers)) if not strategy.recompute else []
                            strategy_details['recompute_layers_count'] = 0
                            strategy_details['no_recompute_layers_count'] = args.total_layers if not strategy.recompute else 0
                            strategy_details['recompute_ratio'] = 0.0
                    
                    # Calculate memory efficiency metrics
                    memory_usage_gb = optimal_solution['memory_cost'][0] / 1024  # Convert MB to GB
                    memory_efficiency = memory_usage_gb / args.memory_upper_limit * 100
                    
                    info = {
                        'optimization_target': 'throughput',
                        'search_timestamp': __import__('datetime').datetime.now().isoformat(),
                        'model_config': {
                            'total_layers': args.total_layers,
                            'world_size': args.world_size,
                            'memory_limit_gb': args.memory_upper_limit,
                            'mixed_precision': args.mixed_precision_type
                        },
                        'optimal_solution': {
                            'batch_size': optimal_solution['bsz'],
                            'accumulation_steps': optimal_solution['accumulation_steps'],
                            'micro_batch_size': optimal_solution['bsz'] // optimal_solution['accumulation_steps'],
                            'global_batch_size': optimal_solution['bsz'],
                        },
                        'strategy': strategy_details,
                        'performance_metrics': {
                            'throughput_samples_per_sec': optimal_solution['throughput'],
                            'time_cost_seconds': optimal_solution['time_cost'],
                            'memory_cost_mb': optimal_solution['memory_cost'],
                            'memory_usage_gb': round(memory_usage_gb, 2),
                            'memory_efficiency_percent': round(memory_efficiency, 1)
                        },
                        'recompute_analysis': self._analyze_recompute_strategy(strategy, args.total_layers) if strategy.recompute else {
                            'enabled': False,
                            'reason': 'Sufficient memory available - no recompute needed for optimal performance'
                        },
                        'optimal_recompute_summary': {
                            'enabled': strategy.recompute > 0,
                            'total_layers': args.total_layers,
                            'recompute_layers_count': strategy_details.get('recompute_layers_count', 0),
                            'recompute_ratio': f"{strategy_details.get('recompute_ratio', 0.0):.1%}",
                            'memory_vs_compute_tradeoff': f"Using {strategy_details.get('recompute_layers_count', 0)} out of {args.total_layers} layers for recompute saves memory at cost of {strategy_details.get('recompute_ratio', 0.0):.1%} additional computation",
                            'efficiency_explanation': 'No recompute - maximum performance' if strategy_details.get('recompute_layers_count', 0) == 0 else f'Minimal recompute strategy - only {strategy_details.get("recompute_layers_count", 0)} layers recompute for optimal memory/performance balance' if strategy_details.get('recompute_layers_count', 0) < args.total_layers else 'Full recompute - maximum memory savings'
                        }
                    }
                else:
                    info = {
                        'error': 'No feasible solution found - all configurations resulted in OOM (Out of Memory)',
                        'suggestion': 'Try reducing batch size, enabling more aggressive recompute, or using smaller model',
                        'search_timestamp': __import__('datetime').datetime.now().isoformat(),
                        'model_config': {
                            'total_layers': args.total_layers,
                            'world_size': args.world_size,
                            'memory_limit_gb': args.memory_upper_limit,
                            'mixed_precision': args.mixed_precision_type
                        }
                    }
                json.dump(info, f, indent=4)
            return results, optimal_solution
        else:
            raise NotImplementedError(f"Search granularity '{args.search_granularity}' is not implemented.")