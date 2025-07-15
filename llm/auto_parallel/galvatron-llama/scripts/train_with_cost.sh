#!/bin/bash

# Training script that uses optimal strategies from Galvatron cost search
# This script reads optimal_solution.json and applies the found strategy

set -e

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "🚀 Training with Galvatron Cost-Optimized Strategy"
echo "   Project root: $PROJECT_ROOT"
echo "   Using improved strategy from cost search algorithm"
echo ""

# 检查 optimal_solution.json 是否存在
SOLUTION_FILE="./configs/optimal_solution.json"
if [ ! -f "$SOLUTION_FILE" ]; then
    echo "❌ Optimal solution file not found: $SOLUTION_FILE"
    echo "   Please run cost search first:"
    echo "   bash scripts/search_fine_grained_recompute.sh"
    exit 1
fi

echo "📊 Found optimal solution file: $SOLUTION_FILE"

# 显示策略摘要
if command -v python3 >/dev/null 2>&1; then
    echo "📋 Strategy Summary:"
    python3 -c "
import json
try:
    with open('$SOLUTION_FILE', 'r') as f:
        data = json.load(f)
    strategy = data.get('strategy', {})
    optimal = data.get('optimal_solution', {})
    performance = data.get('performance_metrics', {})
    
    print(f'   Strategy: {strategy.get(\"strategy_name\", \"Unknown\")}')
    print(f'   Parallel config: PP={strategy.get(\"pp_size\", 1)}, TP={strategy.get(\"tp_size\", 1)}, DP={strategy.get(\"dp_size\", 1)}')
    print(f'   Sharding: Stage {strategy.get(\"sharding_stage\", 0)}')
    print(f'   Recompute: {strategy.get(\"recompute_granularity\", \"None\")}')
    print(f'   Batch size: {optimal.get(\"batch_size\", \"Unknown\")}')
    print(f'   Expected throughput: {performance.get(\"throughput_samples_per_sec\", 0):.2f} samples/s')
    print(f'   Expected memory: {performance.get(\"memory_usage_gb\", 0):.1f}GB')
except Exception as e:
    print(f'   Error reading strategy: {e}')
"
fi

echo ""

# 设置任务名称
TASK_NAME="cost_optimized_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./output/$TASK_NAME"

# 清理输出目录
rm -rf "$OUTPUT_DIR"
rm -rf "${OUTPUT_DIR}_log"

echo "📁 Output directory: $OUTPUT_DIR"
echo ""

# 设置环境变量
export SOT_LOG_LEVEL=4
export PYTHONPATH=../../../:$PYTHONPATH

# 训练脚本配置
TRAINER="./train_with_cost.py"
PYTHON_CMD="/home/pkuhetu/conda_envs/lyx-paddle/bin/python"
LAUNCHER="$PYTHON_CMD -u -m paddle.distributed.launch"

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"} 
GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# 根据GPU数量设置启动器
LAUNCHER="${LAUNCHER} --gpus $GPUS"
LAUNCHER="${LAUNCHER} --log_dir ${OUTPUT_DIR}_log ${TRAINER}"

echo "🔧 Configuration:"
echo "   GPUs: $GPUS (count: $GPU_COUNT)"
echo "   Solution file: $SOLUTION_FILE"
echo "   Output directory: $OUTPUT_DIR"
echo ""

# 基础训练参数 - 简化版本，大部分配置由Python脚本从solution文件读取
TRAIN_ARGS="
    --output_dir $OUTPUT_DIR \
    --solution_file $SOLUTION_FILE \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --learning_rate 3e-05 \
    --min_learning_rate 3e-06 \
    --max_training_steps 10 \
    --logging_steps 1 \
    --do_train true \
    --disable_tqdm true \
    --skip_profile_timer false \
    --skip_memory_metrics 0 \
    --save_total_limit 2 \
    --device gpu \
    --dataloader_num_workers 1 \
    --distributed_dataloader 0 \
    --enable_auto_parallel 1 \
    --autotuner_benchmark false \
    --bf16 true \
    --fp16_opt_level O1 \
    --amp_master_grad false \
    --recompute_use_reentrant true \
    --to_static 0 \
    --virtual_pp_degree 1 \
    --pipeline_schedule_mode 1F1B \
    --sep_parallel_degree 1 \
    --fuse_attention_ffn true \
    --fuse_attention_qkv true \
    --fused_linear_param_grad_add 1 \
    --fuse_sequence_parallel_allreduce true \
    --use_flash_attention true \
    --use_fused_rope true \
    --use_fused_rms_norm true \
"

# 数据配置
DATA_ARGS="
    --input_dir ./data \
    --split 949,50,1 \
    --max_seq_length 1024 \
"

# 性能分析配置
PROFILE_ARGS="
    --profile_time_flag 1 \
    --profile_memory_flag 1 \
    --profile_forward_only 0 \
    --save_time_flag 0 \
    --save_memory_flag 0 \
"

# 调试配置
DEBUG_ARGS="
    --job_schedule_profiler_start 1 \
    --job_schedule_profiler_end 5 \
"

echo "🚀 Starting training with cost-optimized strategy..."
echo ""

# 启动训练
$LAUNCHER \
    $TRAIN_ARGS \
    $MODEL_ARGS \
    $CONFIG_ARGS \
    $PARALLEL_ARGS \
    $OPTIMIZER_ARGS \
    $DATA_ARGS \
    $PROFILE_ARGS \
    $DEBUG_ARGS

TRAIN_EXIT_CODE=$?

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 Training Results:"
    echo "   Output directory: $OUTPUT_DIR"
    echo "   Log directory: ${OUTPUT_DIR}_log"
    echo ""
    echo "📈 Performance Analysis:"
    if [ -f "${OUTPUT_DIR}_log/workerlog.0" ]; then
        echo "   Check detailed logs in: ${OUTPUT_DIR}_log/"
        
        # 尝试提取吞吐量信息
        if grep -q "ips:" "${OUTPUT_DIR}_log/workerlog.0" 2>/dev/null; then
            echo "   Achieved throughput:"
            grep "ips:" "${OUTPUT_DIR}_log/workerlog.0" | tail -5 | while read line; do
                echo "      $line"
            done
        fi
        
        # 尝试提取loss信息
        if grep -q "train_loss" "${OUTPUT_DIR}_log/workerlog.0" 2>/dev/null; then
            echo "   Training loss progression:"
            grep "train_loss" "${OUTPUT_DIR}_log/workerlog.0" | tail -3 | while read line; do
                echo "      $line"
            done
        fi
    fi
    
    echo ""
    echo "🔄 Compare with Expected Performance:"
    python3 -c "
import json
try:
    with open('$SOLUTION_FILE', 'r') as f:
        data = json.load(f)
    performance = data.get('performance_metrics', {})
    expected_throughput = performance.get('throughput_samples_per_sec', 0)
    memory_usage = performance.get('memory_usage_gb', 0)
    
    print(f'   Expected throughput: {expected_throughput:.2f} samples/s')
    print(f'   Expected memory usage: {memory_usage:.1f} GB')
    print(f'   Strategy: {data.get(\"strategy\", {}).get(\"strategy_name\", \"Unknown\")}')
    
    # 显示重计算配置
    strategy = data.get('strategy', {})
    if strategy.get('recompute', 0) == 1:
        recompute_layers = sum(strategy.get('layerwise_recompute', []))
        total_layers = data.get('model_config', {}).get('total_layers', 0)
        print(f'   Recompute: {recompute_layers}/{total_layers} layers')
    else:
        print(f'   Recompute: Disabled')
except Exception as e:
    print(f'   Error reading expected performance: {e}')
"
    
else
    echo "❌ Training failed (exit code: $TRAIN_EXIT_CODE)"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "   1. Check logs in: ${OUTPUT_DIR}_log/"
    echo "   2. Verify optimal solution file: $SOLUTION_FILE"
    echo "   3. Ensure GPU resources match the strategy requirements"
    echo "   4. Check PaddleNLP installation and dependencies"
    echo ""
    echo "🔧 Common Issues:"
    echo "   - Memory OOM: Large vocab_size (32000) may cause cross-entropy OOM"
    echo "   - GPU count mismatch: Ensure GPUS matches world_size in solution"
    echo "   - Missing dependencies: Check PaddleNLP galvatron experimental module"
    
    # 显示最后几行错误日志
    if [ -f "${OUTPUT_DIR}_log/workerlog.0" ]; then
        echo ""
        echo "📋 Last few lines of error log:"
        tail -10 "${OUTPUT_DIR}_log/workerlog.0" 2>/dev/null | while read line; do
            echo "      $line"
        done
    fi
    
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "🎉 Cost-optimized training workflow completed!" 