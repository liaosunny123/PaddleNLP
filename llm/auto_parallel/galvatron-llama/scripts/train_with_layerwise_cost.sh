#!/bin/bash

# Training with Layerwise Cost Optimization Results
# 基于 layerwise 成本优化结果的训练脚本

set -e

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "🚀 Starting training with layerwise cost optimization results..."

# 检查 optimal_solution.json 文件
OPTIMAL_SOLUTION_PATH="./configs/optimal_solution.json"

if [ ! -f "$OPTIMAL_SOLUTION_PATH" ]; then
    echo "❌ Optimal solution file not found: $OPTIMAL_SOLUTION_PATH"
    echo "   Please run layerwise search first:"
    echo "   bash scripts/search_with_galvatron_cost.sh"
    exit 1
fi

echo "✅ Found optimal solution file: $OPTIMAL_SOLUTION_PATH"

# 解析 optimal_solution.json 获取基本配置
if command -v python3 >/dev/null 2>&1; then
    echo "📋 Reading optimal solution configuration..."
    SOLUTION_INFO=$(python3 -c "
import json
try:
    with open('$OPTIMAL_SOLUTION_PATH', 'r') as f:
        data = json.load(f)
    
    strategy = data.get('strategy', {})
    search_metadata = data.get('search_metadata', {})
    
    # Extract configuration
    total_layers = search_metadata.get('total_layers', 16)
    world_size = search_metadata.get('world_size', 8)
    pp_size = strategy.get('pp_size', 1)
    tp_size = strategy.get('tp_size', 1)
    dp_size = strategy.get('dp_size', world_size // (pp_size * tp_size))
    batch_size = data.get('batch_size', 32)
    accumulation_steps = data.get('accumulation_steps', 1)
    
    # Extract recompute configuration
    recompute_flag = strategy.get('recompute', 1)
    recompute_enabled = 'true' if (isinstance(recompute_flag, bool) and recompute_flag) or (isinstance(recompute_flag, int) and recompute_flag > 0) else 'false'
    
    # Calculate per device batch size
    per_device_batch_size = max(1, batch_size // (dp_size * accumulation_steps))
    
    print(f'TOTAL_LAYERS={total_layers}')
    print(f'WORLD_SIZE={world_size}')
    print(f'PP_SIZE={pp_size}')
    print(f'TP_SIZE={tp_size}')
    print(f'DP_SIZE={dp_size}')
    print(f'BATCH_SIZE={batch_size}')
    print(f'ACCUMULATION_STEPS={accumulation_steps}')
    print(f'PER_DEVICE_BATCH_SIZE={per_device_batch_size}')
    print(f'RECOMPUTE_ENABLED={recompute_enabled}')
    
except Exception as e:
    print(f'# Error reading optimal solution: {e}')
    print('TOTAL_LAYERS=16')
    print('WORLD_SIZE=8')
    print('PP_SIZE=1')
    print('TP_SIZE=1')
    print('DP_SIZE=8')
    print('BATCH_SIZE=32')
    print('ACCUMULATION_STEPS=1')
    print('PER_DEVICE_BATCH_SIZE=4')
    print('RECOMPUTE_ENABLED=true')
")
    
    # 解析配置变量
    eval "$SOLUTION_INFO"
    
    echo "   📊 Configuration from optimal solution:"
    echo "      ├─ Total layers: $TOTAL_LAYERS"
    echo "      ├─ World size: $WORLD_SIZE"
    echo "      ├─ Parallelism: PP=$PP_SIZE, TP=$TP_SIZE, DP=$DP_SIZE"
    echo "      ├─ Global batch size: $BATCH_SIZE"
    echo "      ├─ Accumulation steps: $ACCUMULATION_STEPS"
    echo "      ├─ Per device batch size: $PER_DEVICE_BATCH_SIZE"
    echo "      └─ Recompute enabled: $RECOMPUTE_ENABLED"
else
    echo "⚠️  Python3 not found, using default configuration"
    TOTAL_LAYERS=4
    WORLD_SIZE=8
    PP_SIZE=1
    TP_SIZE=1
    DP_SIZE=8
    BATCH_SIZE=32
    ACCUMULATION_STEPS=1
    PER_DEVICE_BATCH_SIZE=4
    RECOMPUTE_ENABLED=true
fi

# 可以通过环境变量覆盖配置
LAYERNUM=${LAYERNUM:-$TOTAL_LAYERS}
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# 确保 GPU 数量与 world_size 匹配
if [ "$GPU_COUNT" -ne "$WORLD_SIZE" ]; then
    echo "⚠️  Warning: GPU count ($GPU_COUNT) doesn't match optimal world_size ($WORLD_SIZE)"
    echo "   Adjusting world size to match available GPUs: $GPU_COUNT"
    WORLD_SIZE=$GPU_COUNT
    # 重新计算 DP 大小
    DP_SIZE=$((WORLD_SIZE / (PP_SIZE * TP_SIZE)))
    PER_DEVICE_BATCH_SIZE=$((BATCH_SIZE / (DP_SIZE * ACCUMULATION_STEPS)))
    PER_DEVICE_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE > 0 ? PER_DEVICE_BATCH_SIZE : 1))
fi

# 任务配置
task_name=${TASK_NAME:-"layerwise_cost_training"}
output_dir="./output/$task_name"
log_dir="${output_dir}_log"

echo "🔧 Training Configuration:"
echo "   Task name: $task_name"
echo "   Model layers: $LAYERNUM"
echo "   GPUs: $GPUS (count: $GPU_COUNT)"
echo "   Output dir: $output_dir"

# 清理之前的输出
rm -rf "$output_dir/"
rm -rf "$log_dir"

# 设置环境变量
export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH
export CUDA_MODULE_LOADING=EAGER # For 4090 cluster

# 脚本配置
TRAINER="./train_with_layerwise_cost.py"
LAUNCHER="python -u -m paddle.distributed.launch"
LAUNCHER="${LAUNCHER} --gpus $GPUS"
LAUNCHER="${LAUNCHER} --log_dir $log_dir $TRAINER --output_dir $output_dir"

# 训练参数配置
TRAIN_ARGS="
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --max_steps 25 \
    --logging_steps 1 \
    --continue_training 0 \
    --do_train true \
    --disable_tqdm true \
    --skip_profile_timer false \
    --skip_memory_metrics 0 \
    --save_total_limit 2 \
    --device gpu \
    --dataloader_num_workers 1 \
    --distributed_dataloader 0 \
    --enable_auto_parallel 1 \
    --use_optimal_solution true \
    --optimal_solution_path $OPTIMAL_SOLUTION_PATH \
"

# 模型参数配置
MODEL_ARGS="
    --model_name_or_path "llama" \
    --num_hidden_layers $LAYERNUM \
    --intermediate_size 11008 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --seq_length 1024 \
    --num_attention_heads 32 \
"

# 配置参数 - 从 optimal solution 中读取具体配置
CONFIG_ARGS="
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION_STEPS \
    --recompute $RECOMPUTE_ENABLED \
    --recompute_use_reentrant true \
    --recompute_granularity full \
    --pp_recompute_interval 0 \
    --bf16 true \
    --fp16_opt_level "O1" \
    --amp_master_grad false \
    --amp_custom_black_list "reduce_sum" "c_softmax_with_cross_entropy" \
    --amp_custom_white_list "lookup_table" "lookup_table_v2" \
"

# 并行参数配置 - 这些将被 optimal solution 覆盖
PARALLEL_ARGS=(
    --to_static 1
    --sharding_parallel_degree 1
    --sharding ""
    --tensor_parallel_degree $TP_SIZE
    --sequence_parallel true
    --pipeline_parallel_degree $PP_SIZE
    --virtual_pp_degree 1
    --pipeline_schedule_mode "1F1B"
    --sep_parallel_degree 1
    --pipeline_parallel_config "enable_send_recv_overlap"
    --data_parallel_config "enable_allreduce_avg_in_gradinent_scale gradient_sync_after_accumulate"
    --sharding_parallel_config "enable_overlap enable_release_grads"
    --tensor_parallel_config "enable_mp_async_allreduce"
)

# 优化器参数配置
DEFAULT_OPTIMIZER_ARGS="
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --fused_linear_param_grad_add 1 \
    --fuse_sequence_parallel_allreduce true \
    --use_flash_attention true \
    --use_fused_rope true \
    --use_fused_rms_norm true \
"

# 数据参数配置
DATA_ARGS="
    --input_dir ./data \
    --split 949,50,1 \
    --max_seq_length 1024"

# 运行时 Profile 参数配置
RUNTIME_PROFILE_ARGS="
    --profile_time_flag 1 \
    --profile_memory_flag 1 \
    --profile_forward_only 0 \
    --save_time_flag 0 \
    --save_memory_flag 0 \
"

# 调试参数配置
DEBUG_ARGS="
    --job_schedule_profiler_start 1 \
    --job_schedule_profiler_end 5 \
"

echo ""
echo "🚀 Launching training with optimal layerwise configuration..."
echo "   Using configuration from: $OPTIMAL_SOLUTION_PATH"
echo "   Command: $LAUNCHER [MODEL_ARGS] [TRAIN_ARGS] [CONFIG_ARGS] [PARALLEL_ARGS] [OPTIMIZER_ARGS] [DATA_ARGS]"
echo ""

# 执行训练
$LAUNCHER \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $CONFIG_ARGS \
    "${PARALLEL_ARGS[@]}" \
    $DEFAULT_OPTIMIZER_ARGS \
    $DATA_ARGS \
    $RUNTIME_PROFILE_ARGS \
    $DEBUG_ARGS

TRAINING_EXIT_CODE=$?

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training with layerwise cost optimization completed successfully!"
    
    echo ""
    echo "📊 Training Results:"
    echo "   Output directory: $output_dir"
    echo "   Log directory: $log_dir"
    
    # 显示最终的内存和性能信息
    if [ -f "$log_dir/workerlog.0" ]; then
        echo ""
        echo "📈 Performance Summary:"
        echo "   (From training logs - last few lines):"
        tail -n 5 "$log_dir/workerlog.0" | grep -E "(memory|allocated|samples|throughput|loss)" || echo "   See full logs in $log_dir/workerlog.0"
    fi
    
    echo ""
    echo "🎯 Optimal Configuration Applied:"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json
try:
    with open('$OPTIMAL_SOLUTION_PATH', 'r') as f:
        data = json.load(f)
    
    strategy = data.get('strategy', {})
    recompute_summary = data.get('optimal_recompute_summary', {})
    
    print(f'   Strategy: {strategy.get(\"strategy_name\", \"Unknown\")}')
    if recompute_summary.get('enabled', False):
        total_layers = recompute_summary.get('total_layers', 0)
        recompute_count = recompute_summary.get('recompute_layers_count', 0)
        print(f'   Recompute: {recompute_count}/{total_layers} layers ({recompute_summary.get(\"recompute_ratio\", \"0%\")})')
        layerwise_pattern = strategy.get('layerwise_recompute', [])
        if layerwise_pattern:
            pattern_str = ''.join(['R' if x == 1 else 'N' for x in layerwise_pattern])
            print(f'   Pattern: {pattern_str} (R=Recompute, N=Normal)')
    else:
        print(f'   Recompute: Disabled (maximum performance mode)')
        
except Exception as e:
    print(f'   Error reading solution: {e}')
"
    fi
    
else
    echo "❌ Training failed (exit code: $TRAINING_EXIT_CODE)"
    echo "   Check logs in: $log_dir"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Verify optimal solution file is valid"
    echo "   2. Check GPU availability and CUDA setup"
    echo "   3. Ensure sufficient memory for the configuration"
    echo "   4. Review training logs for specific errors"
    exit $TRAINING_EXIT_CODE
fi

echo ""
echo "🎉 Layerwise cost-optimized training completed!"
echo ""
echo "💡 Key Features Used:"
echo "   ✨ Optimal parallelism configuration from cost search"
echo "   🧠 Fine-grained layerwise recompute strategy"
echo "   📊 Memory-efficient batch size and accumulation steps"
echo "   🚀 Performance-optimized training setup" 