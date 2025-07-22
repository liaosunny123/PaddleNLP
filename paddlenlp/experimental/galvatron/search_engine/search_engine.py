from ..utils import Strategy
from dataclasses import dataclass, field
from ..cost_model.profile_data_parser import ProfileDataParser, ProfileDataParserArguments
import math
from typing import List
import copy

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
    
    # Layerwise recompute specific arguments
    total_layers: int = field(default=8, metadata={"help": "Total number of transformer layers in the model."})
    enable_layerwise_recompute_search: bool = field(default=False, metadata={"help": "Whether to enable layerwise recompute search."})
    enable_memory_efficiency_mode: bool = field(default=True, metadata={"help": "Whether to prioritize memory efficiency in search."})
    
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
        self.total_layers = int(args_dict.get("--total_layers", self.total_layers))
        self.enable_layerwise_recompute_search = args_dict.get("--enable_layerwise_recompute_search", str(self.enable_layerwise_recompute_search)).lower() == 'true'
        self.enable_memory_efficiency_mode = args_dict.get("--enable_memory_efficiency_mode", str(self.enable_memory_efficiency_mode)).lower() == 'true'
        
class SearchEngine:
    def __init__(self, args_dict):
        self.args = SearchEngineArguments()
        self.args.initialize(args_dict)
        
        parser_data_args = ProfileDataParserArguments()
        parser_data_args.initialize(args_dict)
        self.parser = ProfileDataParser(parser_data_args)
        
        self.generate_strategies()
        self.set_searching_bsz()
        
        if self.args.enable_layerwise_recompute_search:
            print(f'🔧 Layerwise Recompute Search enabled with {self.args.total_layers} layers')
            print(f'   Memory efficiency mode: {self.args.enable_memory_efficiency_mode}')

    def generate_strategies(self):
        args = self.args
        
        self.strategy_set:List[Strategy] = []
        
        i, degree_set = 1, []
        while i <= args.world_size:
            degree_set.append(i)
            i *= 2
        
        for pp_size in degree_set:
            if pp_size > args.max_pp_size:
                continue
            for tp_size in degree_set:
                if pp_size * tp_size > args.world_size:
                    continue
                if tp_size > args.max_tp_size:
                    continue
                dp_size = args.world_size // (pp_size * tp_size)
                # sharding_stage_set = [0, 2, 3] if dp_size > 1 else [0]
                sharding_stage_set = [0, 2] if dp_size > 1 else [0] # when in static mode, RuntimeError: Operation((%0) = "pd_op.embedding_grad" is not support sharded by shard_tensor op in pir mode happend.
                for sharding_stage in sharding_stage_set:
                    # Only generate base strategies without recompute
                    # Layerwise recompute will be handled separately if enabled
                    strategy = Strategy(pp_size=pp_size, tp_size=tp_size, dp_size=dp_size, sharding_stage=sharding_stage, recompute=0)
                    self.strategy_set.append(strategy)
                            
        print(f'SearchEngine strategt_set: {self.strategy_set}')
    
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
    
    def generate_layerwise_recompute_strategies(self, base_strategy: Strategy) -> List[Strategy]:
        """Generate all possible layerwise recompute strategies for a base strategy"""
        strategies = []
        
        if not self.args.enable_layerwise_recompute_search:
            # Only return the base strategy without recompute
            return [base_strategy]
        
        # Generate strategies for different numbers of recompute layers
        # From 0 recompute layers to all layers having recompute
        for recompute_count in range(self.args.total_layers + 1):
            strategy = copy.deepcopy(base_strategy)
            strategy.set_layerwise_recompute(self.args.total_layers, recompute_count)
            strategies.append(strategy)
        
        return strategies
    
    def estimate_layerwise_memory_cost(self, base_strategy: Strategy, layerwise_recompute: List[int], 
                                     global_batch_size: int, mixed_precision_type: str, accumulation_steps: int) -> List[float]:
        """Estimate memory cost for layerwise recompute strategy using original cost models"""
        
        if sum(layerwise_recompute) == 0:
            # No recompute layers - use base strategy with recompute=0
            strategy_no_recompute = copy.deepcopy(base_strategy)
            strategy_no_recompute.recompute = 0
            return self.parser.get_memory_cost_for_specific_strategy(
                strategy_no_recompute, global_batch_size, mixed_precision_type, accumulation_steps)
        
        elif sum(layerwise_recompute) == len(layerwise_recompute):
            # All layers use recompute - use base strategy with recompute=1
            strategy_full_recompute = copy.deepcopy(base_strategy)
            strategy_full_recompute.recompute = 1
            return self.parser.get_memory_cost_for_specific_strategy(
                strategy_full_recompute, global_batch_size, mixed_precision_type, accumulation_steps)
        
        else:
            # Mixed layerwise recompute - calculate based on recompute ratio using original models
            strategy_no_recompute = copy.deepcopy(base_strategy)
            strategy_no_recompute.recompute = 0
            memory_cost_no_recompute = self.parser.get_memory_cost_for_specific_strategy(
                strategy_no_recompute, global_batch_size, mixed_precision_type, accumulation_steps)
            
            strategy_full_recompute = copy.deepcopy(base_strategy)
            strategy_full_recompute.recompute = 1
            memory_cost_full_recompute = self.parser.get_memory_cost_for_specific_strategy(
                strategy_full_recompute, global_batch_size, mixed_precision_type, accumulation_steps)
            
            # Calculate weighted average based on recompute ratio
            recompute_ratio = sum(layerwise_recompute) / len(layerwise_recompute)
            
            adjusted_memory_cost = []
            for i in range(len(memory_cost_no_recompute)):
                # Linear interpolation between no-recompute and full-recompute costs
                adjusted_memory = (memory_cost_no_recompute[i] * (1 - recompute_ratio) + 
                                 memory_cost_full_recompute[i] * recompute_ratio)
                adjusted_memory_cost.append(adjusted_memory)
            
            return adjusted_memory_cost

    def estimate_layerwise_time_cost(self, base_strategy: Strategy, layerwise_recompute: List[int], 
                                   global_batch_size: int, mixed_precision_type: str, accumulation_steps: int) -> float:
        """Estimate time cost for layerwise recompute strategy using original cost models"""
        
        if sum(layerwise_recompute) == 0:
            # No recompute layers - use base strategy with recompute=0
            strategy_no_recompute = copy.deepcopy(base_strategy)
            strategy_no_recompute.recompute = 0
            return self.parser.get_time_cost_for_specific_strategy(
                strategy_no_recompute, global_batch_size, mixed_precision_type, accumulation_steps)
        
        elif sum(layerwise_recompute) == len(layerwise_recompute):
            # All layers use recompute - use base strategy with recompute=1
            strategy_full_recompute = copy.deepcopy(base_strategy)
            strategy_full_recompute.recompute = 1
            return self.parser.get_time_cost_for_specific_strategy(
                strategy_full_recompute, global_batch_size, mixed_precision_type, accumulation_steps)
        
        else:
            # Mixed layerwise recompute - calculate based on recompute ratio using original models
            strategy_no_recompute = copy.deepcopy(base_strategy)
            strategy_no_recompute.recompute = 0
            time_cost_no_recompute = self.parser.get_time_cost_for_specific_strategy(
                strategy_no_recompute, global_batch_size, mixed_precision_type, accumulation_steps)
            
            strategy_full_recompute = copy.deepcopy(base_strategy)
            strategy_full_recompute.recompute = 1
            time_cost_full_recompute = self.parser.get_time_cost_for_specific_strategy(
                strategy_full_recompute, global_batch_size, mixed_precision_type, accumulation_steps)
            
            # Calculate weighted average based on recompute ratio
            recompute_ratio = sum(layerwise_recompute) / len(layerwise_recompute)
            
            # Linear interpolation between no-recompute and full-recompute costs
            adjusted_time_cost = (time_cost_no_recompute * (1 - recompute_ratio) + 
                                time_cost_full_recompute * recompute_ratio)
            
            return adjusted_time_cost
        
    def parallelism_optimization(self):
        args = self.args
        
        if args.search_granularity == 'coarse-grained' or args.search_granularity == 'layerwise':
            optimal_solution, max_throughput, optimal_history = {}, -1, []
            results = dict()
            total_combinations = 0
            evaluated_combinations = 0
            
            for bsz in self.BSZs:
                results[bsz] = dict()
                accumulation_steps_list = range(1, bsz + 1)
                for accumulation_steps in accumulation_steps_list:
                    results[bsz][accumulation_steps] = dict()
                    if bsz % accumulation_steps != 0:
                        continue
                        
                    for base_strategy in self.strategy_set:
                        if bsz // accumulation_steps < base_strategy.dp_size:
                            continue
                        
                        # Generate layerwise recompute strategies if enabled
                        if args.enable_layerwise_recompute_search:
                            strategies_to_evaluate = self.generate_layerwise_recompute_strategies(base_strategy)
                        else:
                            # Traditional approach: evaluate both with and without recompute
                            strategy_no_recompute = copy.deepcopy(base_strategy)
                            strategy_no_recompute.recompute = 0
                            strategy_with_recompute = copy.deepcopy(base_strategy)
                            strategy_with_recompute.recompute = 1
                            strategies_to_evaluate = [strategy_no_recompute, strategy_with_recompute]
                        
                        for strategy in strategies_to_evaluate:
                            total_combinations += 1
                            strategy_key = strategy.serialize()
                            results[bsz][accumulation_steps][strategy_key] = dict()
                            
                            # Calculate memory and time costs
                            if args.enable_layerwise_recompute_search and strategy.layerwise_recompute:
                                memory_cost = self.estimate_layerwise_memory_cost(
                                    base_strategy, strategy.layerwise_recompute, bsz, 
                                    args.mixed_precision_type, accumulation_steps)
                                time_cost = self.estimate_layerwise_time_cost(
                                    base_strategy, strategy.layerwise_recompute, bsz, 
                                    args.mixed_precision_type, accumulation_steps)
                            else:
                                memory_cost = self.parser.get_memory_cost_for_specific_strategy(
                                    strategy, bsz, args.mixed_precision_type, accumulation_steps)
                                time_cost = self.parser.get_time_cost_for_specific_strategy(
                                    strategy, bsz, args.mixed_precision_type, accumulation_steps)
                            
                            evaluated_combinations += 1
                            
                            results[bsz][accumulation_steps][strategy_key]['memory_cost'] = memory_cost
                            results[bsz][accumulation_steps][strategy_key]['time_cost'] = time_cost
                            results[bsz][accumulation_steps][strategy_key]['throughput'] = bsz / time_cost if time_cost > 0 else 0
                            results[bsz][accumulation_steps][strategy_key]['OOM'] = memory_cost[0] > args.memory_upper_limit * 1024 # memory_cost[0] means the first stage memory cost
                            
                            # Check if this is the best solution so far
                            current_throughput = results[bsz][accumulation_steps][strategy_key]['throughput']
                            current_oom = results[bsz][accumulation_steps][strategy_key]['OOM']
                            
                            if current_throughput > max_throughput and not current_oom:
                                max_throughput = current_throughput
                                optimal_solution = {
                                    'bsz': bsz,
                                    'accumulation_steps': accumulation_steps,
                                    'strategy': strategy,
                                    'memory_cost': memory_cost,
                                    'time_cost': time_cost,
                                    'throughput': max_throughput
                                }
                                optimal_history.append(copy.deepcopy(optimal_solution))
                            
                            # Progress reporting
                            if args.enable_layerwise_recompute_search and evaluated_combinations % 50 == 0:
                                recompute_info = strategy.get_recompute_efficiency_info()
                                recompute_desc = f"({recompute_info['recompute_layers_count']}/{recompute_info['total_layers']} layers)" if recompute_info['enabled'] else "(no recompute)"
                                
                                print(f'[{evaluated_combinations:4d}] BSZ: {bsz:2d}, Acc: {accumulation_steps:2d}, '
                                      f'Strategy: {base_strategy.serialize()}, Recompute: {recompute_desc}, '
                                      f'Memory: {memory_cost[0]:.0f}MB, Time: {time_cost:.3f}s, '
                                      f'Throughput: {current_throughput:.2f} samples/s, OOM: {current_oom}')
                            elif not args.enable_layerwise_recompute_search:
                                print(f'Batch Size: {bsz}, Accumulation Steps: {accumulation_steps}, Strategy: {strategy.serialize()}, Memory Cost: {memory_cost} MB, Time Cost: {time_cost} s, Throughput: {current_throughput} Sample/s, OOM: {current_oom}')
            
            # Print optimization results
            if args.enable_layerwise_recompute_search:
                print('\n-----', '[Layerwise Optimization History]', '-----')
                for i, history in enumerate(optimal_history):
                    recompute_info = history['strategy'].get_recompute_efficiency_info()
                    print(f'[{i+1:2d}] BSZ: {history["bsz"]}, Acc: {history["accumulation_steps"]}, '
                          f'Strategy: {history["strategy"].serialize()}, '
                          f'Recompute: {recompute_info["recompute_layers_count"]}/{recompute_info["total_layers"]} layers, '
                          f'Throughput: {history["throughput"]:.2f} samples/s')
                
                print('\n-----', '[Optimal Layerwise Solution]', '-----')
            else:
                print('-----', '[Optimal Solution History]', '-----')
                for history in optimal_history:
                    print(f'Batch Size: {history["bsz"]}, Accumulation Steps: {history["accumulation_steps"]}, Strategy: {history["strategy"].serialize()}, Memory Cost: {history["memory_cost"]} MB, Time Cost: {history["time_cost"]} s, Throughput: {history["throughput"]} Sample/s')
                print('-----', '[Optimal Solution]', '-----')
            
            print('Optimal Solution:', optimal_solution)
            
            # Save optimal solution with enhanced information
            import os 
            current_dir = os.getcwd()
            optimal_solution_path = os.path.join(current_dir, './configs/optimal_solution.json')
            
            if optimal_solution:
                strategy = optimal_solution['strategy']
                
                with open(optimal_solution_path, 'w') as f:
                    import json
                    
                    if args.enable_layerwise_recompute_search:
                        recompute_summary = strategy.get_recompute_efficiency_info()
                        info = {
                            'batch_size': optimal_solution['bsz'],
                            'accumulation_steps': optimal_solution['accumulation_steps'],
                            'strategy': {
                                'strategy_name': strategy.serialize(),
                                'pp_size': strategy.pp_size,
                                'tp_size': strategy.tp_size,
                                'dp_size': strategy.dp_size,
                                'sharding_stage': strategy.sharding_stage,
                                'recompute': strategy.recompute,
                                'layerwise_recompute': strategy.layerwise_recompute,
                                'layerwise_recompute_layers': recompute_summary.get('layerwise_recompute_layers', []),
                                'layerwise_no_recompute_layers': recompute_summary.get('layerwise_no_recompute_layers', [])
                            },
                            'performance_metrics': {
                                'memory_usage_mb': optimal_solution['memory_cost'],
                                'memory_usage_gb': [m/1024 for m in optimal_solution['memory_cost']],
                                'time_cost_seconds': optimal_solution['time_cost'],
                                'throughput_samples_per_sec': optimal_solution['throughput']
                            },
                            'optimal_recompute_summary': recompute_summary,
                            'search_metadata': {
                                'total_layers': args.total_layers,
                                'search_space_size': total_combinations,
                                'evaluated_combinations': evaluated_combinations,
                                'memory_upper_limit_gb': args.memory_upper_limit,
                                'world_size': args.world_size,
                                'enable_layerwise_recompute_search': args.enable_layerwise_recompute_search
                            }
                        }
                    else:
                        info = {
                            'bsz': optimal_solution['bsz'],
                            'accumulation_steps': optimal_solution['accumulation_steps'],
                            'strategy': optimal_solution['strategy'].serialize(),
                            'memory_cost': optimal_solution['memory_cost'],
                            'time_cost': optimal_solution['time_cost'],
                            'throughput': optimal_solution['throughput']
                        }
                    
                    json.dump(info, f, indent=4)
            
            return results, optimal_solution
        else:
            raise NotImplementedError(f"Search granularity '{args.search_granularity}' is not implemented.")