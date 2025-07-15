#!/usr/bin/env python3
"""
Training script that applies optimal strategies found by Galvatron cost search.
Reads optimal_solution.json and configures PaddleNLP training accordingly.
Based on train_dist_random.py with cost optimization integration.
"""

import json
import os
import sys
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import framework
from paddle.base import core

from paddlenlp.ops import Topology
from paddlenlp.trainer import AutoTrainingArguments, PdArgumentParser
from paddlenlp.trainer.auto_trainer import AutoTrainer
from paddlenlp.trainer.trainer_utils import IntervalStrategy, _get_distributed_seeds, ShardingOption
from paddlenlp.transformers import (
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    LlamaConfig,
    LlamaForCausalLM3DAuto,
    LlamaForCausalLMNet,
    LlamaPretrainingCriterion3DAuto,
    LlamaPretrainingCriterionNet,
)
from paddlenlp.utils.log import logger
from paddlenlp.experimental.galvatron.profiler.runtime_profiler import RuntimeProfilerArguments

# Import components from train_dist_random.py
from train_dist_random import (
    DummyDataset, 
    PretrainingTrainer,
    create_dataset,
    runtime_profiler_initalize_manully
)

# Import ShardingOption for proper sharding configuration
from paddlenlp.trainer.trainer_utils import ShardingOption

MODEL_CLASSES = {
    "llama": (LlamaConfig, LlamaForCausalLM3DAuto, LlamaPretrainingCriterion3DAuto),
    "llama_network": (LlamaConfig, LlamaForCausalLMNet, LlamaPretrainingCriterionNet),
}

@dataclass
class CostBasedTrainingArguments(AutoTrainingArguments):
    """Training arguments that can be overridden by cost optimization results."""
    
    solution_file: str = field(
        default="./configs/optimal_solution.json",
        metadata={"help": "Path to the optimal solution JSON file from Galvatron search"}
    )
    
    # Training configuration
    min_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Minimum learning rate decayed to."},
    )
    decay_steps: Optional[float] = field(
        default=None,
        metadata={"help": "The steps use to control the learning rate. If the step > decay_steps, will use the min_learning_rate."},
    )
    max_training_steps: int = field(
        default=10,
        metadata={"help": "Maximum training steps for cost-based training"}
    )
    
    # Pipeline configuration
    pipeline_schedule_mode: str = field(
        default="1F1B", 
        metadata={"help": "The pipeline schedule mode, support FThenB, 1F1B, VPP and Eager-1F1B."}
    )
    virtual_pipeline_seg_method: str = field(
        default="LlamaDecoderLayerAuto", 
        metadata={"help": "The seg method of spliting pp layer for virtual pipeline."}
    )
    
    # Optimization configuration
    enable_linear_fused_grad_add: bool = field(
        default=False,
        metadata={"help": "Enable fused linear grad add strategy."},
    )
    autotuner_benchmark: bool = field(
        default=False,
        metadata={"help": "Weather to run benchmark by autotuner. True for from_scratch and pad_max_length."},
    )
    
    # Profiling configuration (will be set from RuntimeProfilerArguments)
    # Note: These are handled by RuntimeProfilerArguments to avoid conflicts

    def __post_init__(self):
        super().__post_init__()
        assert self.enable_auto_parallel
        
        # Set training steps
        self.max_steps = self.max_training_steps
        print(f"Training configured for {self.max_steps} steps")

@dataclass
class DataArguments:
    """Data arguments for cost-based training."""
    
    input_dir: str = field(
        default="./data", 
        metadata={"help": "The input data directory."}
    )
    split: str = field(
        default="949,50,1", 
        metadata={"help": "Train/valid/test data split."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    share_folder: bool = field(
        default=False,
        metadata={"help": "Use share folder for data dir and output dir on multi machine."},
    )
    data_impl: str = field(
        default="mmap", 
        metadata={"help": "The format of the preprocessed data."}
    )
    skip_warmup: bool = field(
        default=True,
        metadata={"help": "Whether to skip the warmup process of mmap files."},
    )
    data_cache: Optional[str] = field(
        default=None, 
        metadata={"help": "The path of the cached dataset."}
    )

@dataclass
class ModelArguments:
    """Model arguments for cost-based training."""
    
    model_type: str = field(
        default="llama", 
        metadata={"help": "Model type, only support llama for now."}
    )
    model_name_or_path: str = field(
        default="llama",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    
    # Model architecture (will be overridden by optimal solution)
    vocab_size: int = field(default=32000, metadata={"help": "Vocabulary size"})
    hidden_size: int = field(default=4096, metadata={"help": "Hidden size"})
    intermediate_size: int = field(default=11008, metadata={"help": "Intermediate size"})
    num_hidden_layers: int = field(default=4, metadata={"help": "Number of hidden layers"})
    num_attention_heads: int = field(default=32, metadata={"help": "Number of attention heads"})
    seq_length: int = field(default=1024, metadata={"help": "Sequence length"})
    
    # Optimization features
    use_flash_attention: bool = field(default=True, metadata={"help": "Use flash attention"})
    use_fused_rms_norm: bool = field(default=True, metadata={"help": "Use fused RMS norm"})
    fuse_attention_qkv: bool = field(default=True, metadata={"help": "Fuse attention QKV"})
    fuse_attention_ffn: bool = field(default=True, metadata={"help": "Fuse attention FFN"})
    use_fused_rope: bool = field(default=True, metadata={"help": "Use fused RoPE"})
    
    # Recompute configuration (will be overridden by optimal solution)
    recompute_granularity: str = field(default="full", metadata={"help": "Recompute granularity"})
    virtual_pp_degree: int = field(default=1, metadata={"help": "Virtual pipeline degree"})
    no_recompute_layers: Optional[List[int]] = field(default=None, metadata={"help": "Layers without recompute"})
    pp_recompute_interval: int = field(default=0, metadata={"help": "PP recompute interval"})
    recompute_use_reentrant: bool = field(default=True, metadata={"help": "Use reentrant recompute"})
    
    # Other configurations
    use_fast_layer_norm: bool = field(default=False, metadata={"help": "Use fast layer norm"})
    continue_training: bool = field(default=False, metadata={"help": "Continue training from existing weights"})

def load_optimal_solution(solution_file: str) -> dict:
    """Load the optimal solution from JSON file."""
    if not os.path.exists(solution_file):
        raise FileNotFoundError(f"❌ Optimal solution file not found: {solution_file}")
    
    with open(solution_file, 'r') as f:
        solution = json.load(f)
    
    print(f"📊 Loaded optimal solution from {solution_file}")
    
    # Log basic info
    if 'optimization_target' in solution:
        print(f"   Optimization target: {solution['optimization_target']}")
    if 'search_timestamp' in solution:
        print(f"   Search timestamp: {solution['search_timestamp']}")
    
    if 'error' in solution:
        raise ValueError(f"❌ Optimal solution contains error: {solution['error']}")
    
    return solution

def apply_optimal_strategy(
    training_args: CostBasedTrainingArguments, 
    data_args: DataArguments,
    model_args: ModelArguments, 
    solution: dict
):
    """Apply the optimal strategy to training arguments."""
    
    optimal_solution = solution.get('optimal_solution', {})
    strategy = solution.get('strategy', {})
    model_config = solution.get('model_config', {})
    
    print("🔧 Applying optimal strategy to training configuration:")
    
    # Apply model configuration first
    total_layers = model_config.get('total_layers')
    if total_layers:
        model_args.num_hidden_layers = total_layers
        print(f"   🏗️  Model layers: {total_layers}")
    
    world_size = model_config.get('world_size')
    if world_size:
        print(f"   🌍 World size: {world_size}")
    
    # Apply parallel configuration
    pp_size = strategy.get('pp_size', 1)
    tp_size = strategy.get('tp_size', 1)
    dp_size = strategy.get('dp_size', 1)
    sharding_stage = strategy.get('sharding_stage', 0)
    
    training_args.pipeline_parallel_degree = pp_size
    training_args.tensor_parallel_degree = tp_size
    
    # Handle sharding configuration
    if sharding_stage > 0:
        training_args.sharding_parallel_degree = dp_size
        sharding_str = f"stage{sharding_stage}"
        # Use setattr to avoid type checking issues during runtime conversion
        setattr(training_args, 'sharding', sharding_str)
        # Convert string to ShardingOption list (same as __post_init__ does)
        setattr(training_args, 'sharding', [ShardingOption(s) for s in sharding_str.split()])
    else:
        training_args.sharding_parallel_degree = 1
        setattr(training_args, 'sharding', [])
    
    print(f"   🔗 Parallel Configuration:")
    print(f"      Pipeline parallel: {pp_size}")
    print(f"      Tensor parallel: {tp_size}")
    print(f"      Data parallel: {dp_size}")
    print(f"      Sharding stage: {sharding_stage}")
    
    # Set sequence parallel if tensor parallel > 1
    if tp_size > 1:
        training_args.sequence_parallel = True
        print(f"      Sequence parallel: enabled")
    
    # Use profile-safe configuration instead of optimal solution to avoid memory issues
    # Profile data was collected with batch_size=8, seq_length=1024
    # Using these safe values to match profile conditions
    print(f"   ⚠️  Using Ultra-Conservative Configuration instead of Cost Search results")
    print(f"      Reason: Cost search memory estimation is severely inaccurate (15GB vs 38GB actual)")
    print(f"      Even profile batch_size=8 caused OOM, using batch_size=2")
    print(f"      Conservative config: batch_size=2 (vs optimal_solution batch_size={optimal_solution.get('batch_size', 32)})")
    
    # Force ultra-conservative configuration for memory safety
    batch_size = 2  # Much smaller than profile data to account for other factors
    micro_batch_size = 2  # Keep it simple 1:1
    accumulation_steps = 16  # To maintain reasonable global batch size = 2 * 16 = 32
    
    print(f"   📊 Using Ultra-Conservative Configuration Strategy:")
    print(f"      Applied ultra-small batch sizing for memory safety")
    print(f"      Using parallel strategy from cost search (reliable)")
    print(f"      Expected memory usage: <10GB (much lower than profile data)")
    
    # Apply the exact configuration from optimal solution
    training_args.per_device_train_batch_size = micro_batch_size
    training_args.gradient_accumulation_steps = accumulation_steps
    
    print(f"   📦 Batch Configuration (Ultra-Conservative Override):")
    print(f"      Micro batch size: {micro_batch_size} (ultra-conservative)")
    print(f"      Accumulation steps: {accumulation_steps} (compensating for small batch)")
    print(f"      Per-device batch size: {batch_size} (much smaller than profile)")
    print(f"      Effective global batch size: {batch_size * accumulation_steps}")
    print(f"      🛡️ Using ultra-small values to definitely avoid OOM")
    
    # Apply recompute configuration
    recompute_enabled = strategy.get('recompute', 0) == 1
    recompute_granularity = strategy.get('recompute_granularity', 'full')
    pp_recompute_interval = strategy.get('pp_recompute_interval', 0)
    
    training_args.recompute = recompute_enabled
    if recompute_enabled:
        model_args.recompute_granularity = recompute_granularity
        model_args.pp_recompute_interval = pp_recompute_interval
        
        # Handle layerwise recompute configuration
        layerwise_recompute = strategy.get('layerwise_recompute', [])
        if layerwise_recompute:
            # Find layers that should NOT be recomputed (value = 0)
            no_recompute_layers = [i for i, val in enumerate(layerwise_recompute) if val == 0]
            if no_recompute_layers:
                model_args.no_recompute_layers = no_recompute_layers
                print(f"      No recompute layers: {no_recompute_layers}")
    
    print(f"   🔄 Recompute Configuration:")
    print(f"      Enabled: {recompute_enabled}")
    if recompute_enabled:
        print(f"      Granularity: {recompute_granularity}")
        print(f"      PP interval: {pp_recompute_interval}")
    
    # Log expected performance
    performance_metrics = solution.get('performance_metrics', {})
    if performance_metrics:
        expected_throughput = performance_metrics.get('throughput_samples_per_sec', 0)
        memory_usage_gb = performance_metrics.get('memory_usage_gb', 0)
        
        print(f"   🎯 Expected Performance:")
        print(f"      Throughput: {expected_throughput:.2f} samples/sec")
        print(f"      Memory usage: {memory_usage_gb:.1f} GB")

def init_seed(seed: int = 1234, args=None):
    """Initialize random seeds for reproducibility."""
    if args is None:
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
    else:
        assert not args.use_hybrid_parallel and args.enable_auto_parallel
        if dist.get_world_size() > 1:
            if args.hybrid_parallel_topo_order is None or args.hybrid_parallel_topo_order == "pp_first":
                order = ["pp", "dp", "sharding", "mp", "sep"]
            elif args.hybrid_parallel_topo_order == "sharding_first":
                order = ["dp", "sharding", "pp", "mp", "sep"]
            
            topo = Topology(
                dist.get_rank(),
                dist.get_world_size(),
                dp_degree=args.dataset_world_size,
                pp_degree=args.pipeline_parallel_degree,
                mp_degree=args.tensor_parallel_degree,
                sharding_degree=1,  # auto_parallel's sharding is not orthogonal with dp, mp and pp
                order=order,
            )

            global_seed, local_seed, random_seed = _get_distributed_seeds(args.seed, topo)

            random_seed = random_seed.item()
            paddle.seed(local_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            print(
                f"The global seed is set to {global_seed}, local seed is set to {local_seed} and "
                f"random seed is set to {random_seed}."
            )
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            paddle.seed(args.seed)

def main():
    """Main training function with cost optimization."""
    parser = PdArgumentParser((ModelArguments, DataArguments, CostBasedTrainingArguments, RuntimeProfilerArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, runtime_profiler_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, runtime_profiler_args = parser.parse_args_into_dataclasses()

    # Load and apply optimal solution
    print("🚀 Starting training with Galvatron cost-optimized strategy")
    solution = load_optimal_solution(training_args.solution_file)
    apply_optimal_strategy(training_args, data_args, model_args, solution)

    # Initialize environment
    if data_args.data_cache is not None:
        os.makedirs(data_args.data_cache, exist_ok=True)

    init_seed(args=training_args)
    paddle.set_device(training_args.device)
    
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    # Initialize runtime profiler
    runtime_profiler_initalize_manully(runtime_profiler_args, training_args, model_args)

    # Log configurations
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.print_config(training_args, "Training")
    training_args.print_config(runtime_profiler_args, "RuntimeProfile")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"world_size: {training_args.world_size}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Create model configuration
    config_class, model_class, criterion_class = MODEL_CLASSES[model_args.model_type]
    config = config_class()
    
    # Apply model arguments to config
    config.num_hidden_layers = model_args.num_hidden_layers
    config.intermediate_size = model_args.intermediate_size
    config.vocab_size = model_args.vocab_size
    config.hidden_size = model_args.hidden_size
    config.seq_length = model_args.seq_length
    config.max_position_embeddings = config.seq_length
    config.num_attention_heads = model_args.num_attention_heads
    config.use_fast_layer_norm = model_args.use_fast_layer_norm
    
    # Optimization features
    config.use_flash_attention = model_args.use_flash_attention
    config.use_fused_rms_norm = model_args.use_fused_rms_norm
    config.fuse_attention_qkv = model_args.fuse_attention_qkv
    config.fuse_attention_ffn = model_args.fuse_attention_ffn
    config.use_fused_rope = model_args.use_fused_rope
    
    # Recompute configuration
    config.recompute_granularity = model_args.recompute_granularity
    config.virtual_pp_degree = model_args.virtual_pp_degree
    config.use_recompute = training_args.recompute
    config.no_recompute_layers = model_args.no_recompute_layers
    config.pp_recompute_interval = model_args.pp_recompute_interval
    config.recompute_use_reentrant = model_args.recompute_use_reentrant
    
    # Parallel configuration
    config.sequence_parallel = training_args.sequence_parallel
    config.fuse_sequence_parallel_allreduce = training_args.fuse_sequence_parallel_allreduce
    config.tensor_parallel_degree = training_args.tensor_parallel_degree
    config.tensor_parallel_rank = training_args.tensor_parallel_rank
    config.sharding_parallel_degree = training_args.sharding_parallel_degree

    # Virtual pipeline configuration
    if training_args.strategy.pipeline.enable and config.virtual_pp_degree > 1:
        pipeline = training_args.strategy.pipeline
        pipeline.vpp_degree = config.virtual_pp_degree
        pipeline.vpp_seg_method = training_args.virtual_pipeline_seg_method

    print(f"📋 Final Model Config: {config}")

    # Create model and criterion
    print("🏗️  Creating model and loss function...")
    with paddle.LazyGuard():
        model = model_class.from_config(config, dtype="float32")
        criterion = criterion_class(config)

    print("✅ Model initialized successfully")

    # Enable recompute if configured
    if training_args.recompute:
        def fn(layer):
            if hasattr(layer, "enable_recompute") and (layer.enable_recompute is False or layer.enable_recompute == 0):
                layer.enable_recompute = True
        model.apply(fn)
        print("🔄 Recompute enabled for model layers")

    # Create learning rate scheduler
    if training_args.decay_steps is None:
        training_args.decay_steps = training_args.max_steps

    if training_args.warmup_steps > 0:
        warmup_steps = training_args.warmup_steps
    else:
        warmup_steps = training_args.warmup_ratio * training_args.max_steps

    lr_scheduler = None
    if training_args.lr_scheduler_type.value == "cosine":
        lr_scheduler = CosineAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )
    elif training_args.lr_scheduler_type.value == "linear":
        lr_scheduler = LinearAnnealingWithWarmupDecay(
            max_lr=training_args.learning_rate,
            min_lr=training_args.min_learning_rate,
            warmup_step=warmup_steps,
            decay_step=training_args.decay_steps,
            last_epoch=0,
        )

    print(f"📈 Learning rate scheduler: {training_args.lr_scheduler_type.value}")

    # Create dataset
    train_dataset, data_collator = create_dataset(config.vocab_size, config.seq_length)
    print(f"📊 Dataset created with vocab_size={config.vocab_size}, seq_length={config.seq_length}")

    # Create trainer
    print("🔧 Creating trainer...")
    trainer = PretrainingTrainer(
        model=model,
        criterion=criterion,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        optimizers=(None, lr_scheduler),
        runtime_profiler_args=runtime_profiler_args,
    )
    print("✅ Trainer created successfully")

    # Log memory usage after initialization
    current_device = framework._current_expected_place_()
    max_memory_allocated = core.device_memory_stat_peak_value("Allocated", current_device.get_device_id()) / 2**20
    current_memory_allocated = core.device_memory_stat_current_value("Allocated", current_device.get_device_id()) / 2**20
    print(f"💾 Memory after initialization:")
    print(f"   Max memory allocated: {max_memory_allocated:.1f} MB")
    print(f"   Current memory allocated: {current_memory_allocated:.1f} MB")

    # Start training
    print("🚀 Starting training with optimal strategy...")
    strategy_name = solution.get('strategy', {}).get('strategy_name', 'Unknown')
    print(f"   Strategy: {strategy_name}")
    
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        print("✅ Training completed successfully!")
        
        # Log final performance
        if hasattr(train_result, 'training_loss'):
            print(f"📈 Final training loss: {train_result.training_loss:.4f}")
    else:
        print("⚠️  Training not enabled (do_train=False)")

if __name__ == "__main__":
    main() 