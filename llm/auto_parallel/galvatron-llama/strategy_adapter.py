#!/usr/bin/env python3
"""
Strategy Adapter for Galvatron-PaddleNLP Integration
====================================================

This module converts Galvatron's optimal strategy into PaddleNLP training configurations,
supporting fine-grained recompute features.
"""

import json
import argparse
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class PaddleNLPConfig:
    """PaddleNLP training configuration"""
    # Parallel strategy
    tensor_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    dataset_world_size: int = 8  # dp_size
    sharding_parallel_degree: int = 1
    sharding: str = "stage0"
    
    # Batch size configuration
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # Recompute configuration
    recompute: bool = False
    recompute_granularity: str = "full"
    no_recompute_layers: Optional[List[int]] = None
    pp_recompute_interval: int = 0
    
    # Other training parameters
    max_steps: int = 1000
    learning_rate: float = 3e-5
    bf16: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None:
                result[field_name] = field_value
        return result


class StrategyAdapter:
    """Converts Galvatron strategy to PaddleNLP configuration"""
    
    def __init__(self, strategy_file: str = "./configs/optimal_solution.json"):
        self.strategy_file = strategy_file
        self.optimal_strategy = self._load_optimal_strategy()
    
    def _load_optimal_strategy(self) -> Dict[str, Any]:
        """Load optimal strategy from Galvatron search results"""
        try:
            with open(self.strategy_file, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded optimal strategy from {self.strategy_file}")
            return data
        except FileNotFoundError:
            print(f"❌ Strategy file not found: {self.strategy_file}")
            raise
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in strategy file: {self.strategy_file}")
            raise
    
    def convert_to_paddlenlp_config(self) -> PaddleNLPConfig:
        """Convert Galvatron strategy to PaddleNLP configuration"""
        strategy_dict = self.optimal_strategy['strategy']
        
        # Parse strategy string (format: pp{N}_tp{N}_dp{N}_sharding{N}_recompute{N})
        strategy_parts = strategy_dict.split('_')
        
        # Extract parallel degrees
        pp_degree = int([p for p in strategy_parts if p.startswith('pp')][0][2:])
        tp_degree = int([p for p in strategy_parts if p.startswith('tp')][0][2:])
        dp_degree = int([p for p in strategy_parts if p.startswith('dp')][0][2:])
        sharding_stage = int([p for p in strategy_parts if p.startswith('sharding')][0][8:])
        recompute_flag = int([p for p in strategy_parts if p.startswith('recompute')][0][9:])
        
        # Convert sharding stage to PaddleNLP format
        sharding_map = {0: "stage0", 2: "stage2", 3: "stage3"}
        sharding_str = sharding_map.get(sharding_stage, "stage0")
        
        # Calculate batch sizes
        global_batch_size = self.optimal_strategy['bsz']
        accumulation_steps = self.optimal_strategy['accumulation_steps']
        per_device_batch_size = global_batch_size // dp_degree // accumulation_steps
        
        # Create PaddleNLP configuration
        config = PaddleNLPConfig(
            tensor_parallel_degree=tp_degree,
            pipeline_parallel_degree=pp_degree,
            dataset_world_size=dp_degree,
            sharding_parallel_degree=1,  # Auto-parallel uses orthogonal sharding
            sharding=sharding_str,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=accumulation_steps,
            recompute=bool(recompute_flag)
        )
        
        # Handle fine-grained recompute if present
        if recompute_flag and 'recompute_granularity' in strategy_dict:
            config.recompute_granularity = strategy_dict.get('recompute_granularity', 'full')
        
        if recompute_flag and 'no_recompute_layers' in strategy_dict:
            config.no_recompute_layers = strategy_dict.get('no_recompute_layers', [])
            
        if recompute_flag and 'pp_recompute_interval' in strategy_dict:
            config.pp_recompute_interval = strategy_dict.get('pp_recompute_interval', 0)
        
        return config
    
    def generate_training_script(self, config: PaddleNLPConfig, output_file: str = "train_with_galvatron_strategy.py"):
        """Generate PaddleNLP training script with optimal strategy"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated PaddleNLP Training Script with Galvatron Optimal Strategy
=======================================================================

This script was automatically generated by Galvatron Strategy Adapter.
It uses the optimal parallel strategy found by Galvatron search engine.

Strategy Summary:
- Tensor Parallel: {config.tensor_parallel_degree}
- Pipeline Parallel: {config.pipeline_parallel_degree} 
- Data Parallel: {config.dataset_world_size}
- Sharding: {config.sharding}
- Recompute: {config.recompute}
- Batch Size: {config.per_device_train_batch_size} per device
- Accumulation Steps: {config.gradient_accumulation_steps}
"""

import os
import sys

# Add PaddleNLP to path
sys.path.insert(0, "../../../")

from train_dist_random_fine_grained import main

if __name__ == "__main__":
    # Override default arguments with Galvatron optimal strategy
    import sys
    
    # Strategy arguments
    strategy_args = [
        "--tensor_parallel_degree", "{config.tensor_parallel_degree}",
        "--pipeline_parallel_degree", "{config.pipeline_parallel_degree}",
        "--dataset_world_size", "{config.dataset_world_size}",  # This is dp_size
        "--sharding_parallel_degree", "{config.sharding_parallel_degree}",
        "--sharding", "{config.sharding}",
        "--per_device_train_batch_size", "{config.per_device_train_batch_size}",
        "--gradient_accumulation_steps", "{config.gradient_accumulation_steps}",
        "--recompute", "{str(config.recompute).lower()}",
    ]
    
    # Fine-grained recompute arguments
    if config.recompute:
        strategy_args.extend([
            "--recompute_granularity", "{config.recompute_granularity}",
        ])
        
        if config.no_recompute_layers:
            no_recompute_str = ",".join(map(str, config.no_recompute_layers))
            strategy_args.extend([
                "--no_recompute_layers", no_recompute_str,
            ])
            
        if config.pp_recompute_interval > 0:
            strategy_args.extend([
                "--pp_recompute_interval", "{config.pp_recompute_interval}",
            ])
    
    # Default model and training arguments
    default_args = [
        "--model_name_or_path", "llama",
        "--tokenizer_name_or_path", "llama", 
        "--num_hidden_layers", "16",
        "--intermediate_size", "11008",
        "--vocab_size", "32000",
        "--hidden_size", "4096",
        "--seq_length", "1024",
        "--num_attention_heads", "32",
        "--max_steps", "{config.max_steps}",
        "--learning_rate", "{config.learning_rate}",
        "--bf16", "{str(config.bf16).lower()}",
        "--enable_auto_parallel", "1",
        "--do_train", "true",
        "--output_dir", "./output_galvatron",
        "--logging_steps", "10",
        "--save_steps", "500",
    ]
    
    # Combine all arguments
    sys.argv = [sys.argv[0]] + strategy_args + default_args
    
    print("🚀 Starting PaddleNLP training with Galvatron optimal strategy...")
    print(f"   Strategy: TP={config.tensor_parallel_degree}, PP={config.pipeline_parallel_degree}, DP={config.dataset_world_size}")
    print(f"   Recompute: {config.recompute} (granularity: {config.recompute_granularity})")
    print(f"   Batch size: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} × {config.dataset_world_size} = {{config.per_device_train_batch_size * config.gradient_accumulation_steps * config.dataset_world_size}}")
    print()
    
    # Run training
    main()
'''
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(output_file, 0o755)
        
        print(f"✅ Generated training script: {output_file}")
        return output_file
    
    def save_config_json(self, config: PaddleNLPConfig, output_file: str = "galvatron_paddlenlp_config.json"):
        """Save configuration as JSON for reference"""
        config_dict = config.to_dict()
        config_dict['metadata'] = {
            'source': 'Galvatron Strategy Adapter',
            'original_strategy': self.optimal_strategy,
            'note': 'Auto-generated PaddleNLP configuration from Galvatron optimal strategy'
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Saved configuration: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description="Convert Galvatron strategy to PaddleNLP configuration")
    parser.add_argument("--strategy_file", default="./configs/optimal_solution.json",
                       help="Path to Galvatron optimal strategy file")
    parser.add_argument("--output_script", default="train_with_galvatron_strategy.py",
                       help="Output training script filename")
    parser.add_argument("--output_config", default="galvatron_paddlenlp_config.json",
                       help="Output configuration JSON filename")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    try:
        # Create adapter and convert strategy
        adapter = StrategyAdapter(args.strategy_file)
        config = adapter.convert_to_paddlenlp_config()
        
        # Override with command line arguments
        config.max_steps = args.max_steps
        config.learning_rate = args.learning_rate
        
        print("🔄 Converting Galvatron strategy to PaddleNLP configuration...")
        print(f"📊 Optimal Strategy Summary:")
        print(f"   Throughput: {adapter.optimal_strategy['throughput']:.2f} samples/s")
        print(f"   Memory Cost: {adapter.optimal_strategy['memory_cost']} MB")
        print(f"   Time Cost: {adapter.optimal_strategy['time_cost']:.2f} s")
        print()
        
        print(f"⚙️  PaddleNLP Configuration:")
        print(f"   Tensor Parallel: {config.tensor_parallel_degree}")
        print(f"   Pipeline Parallel: {config.pipeline_parallel_degree}")
        print(f"   Data Parallel: {config.dataset_world_size}")
        print(f"   Sharding: {config.sharding}")
        print(f"   Recompute: {config.recompute}")
        if config.recompute:
            print(f"   Recompute Granularity: {config.recompute_granularity}")
            if config.no_recompute_layers:
                print(f"   No Recompute Layers: {config.no_recompute_layers}")
            if config.pp_recompute_interval > 0:
                print(f"   PP Recompute Interval: {config.pp_recompute_interval}")
        print(f"   Batch Size: {config.per_device_train_batch_size} per device")
        print(f"   Accumulation Steps: {config.gradient_accumulation_steps}")
        print()
        
        # Generate outputs
        script_file = adapter.generate_training_script(config, args.output_script)
        config_file = adapter.save_config_json(config, args.output_config)
        
        print("🎉 Strategy conversion completed successfully!")
        print(f"📝 Next steps:")
        print(f"   1. Review configuration: {config_file}")
        print(f"   2. Run training: python {script_file}")
        print(f"   3. Or use distributed training: python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 {script_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 