#!/bin/bash

# Fine-grained Recompute Search for Galvatron-LLaMA (Fixed Version)
# 使用现有的 profile 数据进行细粒度重计算策略搜索

set -e

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查必要的配置文件
check_file() {
    local file=$1
    local name=$2
    
    if [ -f "$file" ] && [ -s "$file" ]; then
        echo "   ✅ $name: $file"
        return 0
    else
        echo "   ❌ $name: Missing or empty - $file"
        return 1
    fi
}

echo "📂 Checking profile data files..."
REQUIRED_FILES=(
    "./configs/computation_profiling_bf16_llama_rank[0].json:Computation Profile"
    "./configs/memory_profiling_bf16_llama.json:Memory Profile"
    "./configs/allreduce_bandwidth_1nodes_8gpus_per_node.json:AllReduce Bandwidth"
    "./configs/p2p_bandwidth_1nodes_8gpus_per_node.json:P2P Bandwidth"
    "./configs/overlap_coefficient.json:Overlap Coefficient"
)

ALL_FILES_EXIST=true
for file_info in "${REQUIRED_FILES[@]}"; do
    IFS=':' read -r file name <<< "$file_info"
    if ! check_file "$file" "$name"; then
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    echo ""
    echo "❌ Missing required profile files. Please run profile collection first:"
    echo "   bash scripts/run_profile.sh"
    exit 1
fi

echo ""
echo "🚀 Starting optimized fine-grained recompute strategy search..."

# 搜索参数配置
LAYERNUM=${LAYERNUM:-8}
TOTAL_LAYERS=${TOTAL_LAYERS:-$LAYERNUM}

# GPU配置 - 支持指定GPU列表，确保world_size与实际GPU数量匹配
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l)
WORLD_SIZE=$GPU_COUNT

# 重计算配置
ONLY_FULL_RECOMPUTE=${ONLY_FULL_RECOMPUTE:-true}
ENABLE_LAYERWISE_RECOMPUTE=${ENABLE_LAYERWISE_RECOMPUTE:-true}
ENABLE_MEMORY_EFFICIENCY_MODE=${ENABLE_MEMORY_EFFICIENCY_MODE:-true}

echo "🔧 Optimized Configuration:"
echo "   Model layers: $LAYERNUM"
echo "   Total layers for search: $TOTAL_LAYERS"
echo "   GPUs: $GPUS (count: $GPU_COUNT)"
echo "   World size: $WORLD_SIZE"
echo "   Only full recompute: $ONLY_FULL_RECOMPUTE"
echo "   Enable layerwise analysis: $ENABLE_LAYERWISE_RECOMPUTE"
echo "   Memory efficiency mode: $ENABLE_MEMORY_EFFICIENCY_MODE"
echo ""

MIN_BSZ=8
MAX_BSZ=32
BSZ_STEP=8
MEMORY_LIMIT=18

echo "   Batch size range: $MIN_BSZ - $MAX_BSZ (step: $BSZ_STEP)"
echo "   Memory limit: ${MEMORY_LIMIT}GB"

# 根据GPU数量调整并行度上限
MAX_TP_SIZE=${MAX_TP_SIZE:-$WORLD_SIZE}
MAX_PP_SIZE=${MAX_PP_SIZE:-$WORLD_SIZE}

# 根据配置设置重计算参数
if [ "$ONLY_FULL_RECOMPUTE" = "true" ]; then
    RECOMPUTE_GRANULARITIES="full"
else
    RECOMPUTE_GRANULARITIES="full"
fi

SEARCH_CONFIG="
--model_name llama_fine_grained
--layernum $LAYERNUM
--hidden_size 4096
--global_batch_size $MAX_BSZ
--world_size $WORLD_SIZE
--min_bsz $MIN_BSZ
--max_bsz $MAX_BSZ
--bsz_step $BSZ_STEP
--max_tp_size $MAX_TP_SIZE
--max_pp_size $MAX_PP_SIZE
--memory_upper_limit $MEMORY_LIMIT
--mixed_precision_type bf16
--enable_fine_grained_recompute_search true
--total_layers $TOTAL_LAYERS
--only_full_recompute $ONLY_FULL_RECOMPUTE
--enable_layerwise_recompute $ENABLE_LAYERWISE_RECOMPUTE
--enable_memory_efficiency_mode $ENABLE_MEMORY_EFFICIENCY_MODE
--time_profile_mode batch
--time_profile_data_path ./configs/computation_profiling_bf16_llama_rank[0].json
--memory_profile_mode static
--memory_profile_data_path ./configs/memory_profiling_bf16_llama.json
--layernum_list $LAYERNUM
--hidden_size_list 4096
--seqlen_list 1024
--overlap_coe_path ./configs/overlap_coefficient.json
--allreduce_coe_path ./configs/allreduce_bandwidth_1nodes_8gpus_per_node.json
--p2p_coe_path ./configs/p2p_bandwidth_1nodes_8gpus_per_node.json
"

# 设置GPU环境变量并运行搜索
echo "🔬 Running optimized Galvatron search engine..."
echo "   Using GPUs: $GPUS"
export CUDA_VISIBLE_DEVICES="$GPUS"

# 设置Python路径
PADDLENLP_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
export PYTHONPATH="$PADDLENLP_ROOT:$PYTHONPATH"
echo "   PaddleNLP path: $PADDLENLP_ROOT"

python search_dist.py $SEARCH_CONFIG

SEARCH_EXIT_CODE=$?

echo ""
if [ $SEARCH_EXIT_CODE -eq 0 ]; then
    echo "✅ Optimized recompute search completed successfully!"
    
    # 检查结果
    if [ -f "./configs/optimal_solution.json" ]; then
        echo ""
        echo "📊 Search Results:"
        echo "   Optimal solution saved to: ./configs/optimal_solution.json"
        
        # 简化的结果显示
        if command -v python3 >/dev/null 2>&1; then
            echo ""
            echo "📋 Optimal Strategy Summary:"
            python3 -c "
import json
try:
    with open('./configs/optimal_solution.json', 'r') as f:
        data = json.load(f)
    
    strategy = data.get('strategy', {})
    performance = data.get('performance_metrics', {})
    optimal_sol = data.get('optimal_solution', {})
    recompute_summary = data.get('optimal_recompute_summary', {})
    
    print(f'   Strategy: {strategy.get(\"strategy_name\", \"Unknown\")}')
    print(f'   Parallelism: PP={strategy.get(\"pp_size\", 1)}, TP={strategy.get(\"tp_size\", 1)}, DP={strategy.get(\"dp_size\", 1)}')
    print(f'   Sharding: Stage {strategy.get(\"sharding_stage\", 0)}')
    print(f'   Batch Size: {optimal_sol.get(\"batch_size\", \"Unknown\")}')
    print(f'   Accumulation Steps: {optimal_sol.get(\"accumulation_steps\", \"Unknown\")}')
    print(f'   Throughput: {performance.get(\"throughput_samples_per_sec\", 0):.2f} samples/s')
    print(f'   Memory: {performance.get(\"memory_usage_gb\", 0):.1f}GB')
    
    # 重计算信息
    if recompute_summary.get('enabled', False):
        recompute_count = recompute_summary.get('recompute_layers_count', 0)
        total_layers = recompute_summary.get('total_layers', 0)
        print(f'   Recompute: {recompute_count}/{total_layers} layers ({recompute_summary.get(\"recompute_ratio\", \"0%\")})')
        print(f'   Strategy: {recompute_summary.get(\"efficiency_explanation\", \"Unknown\")}')
        
        # 显示具体的layerwise pattern
        layerwise_pattern = strategy.get('layerwise_recompute', [])
        if layerwise_pattern:
            print(f'   Layerwise pattern: {layerwise_pattern}')
            print(f'   Recompute layers: {strategy.get(\"layerwise_recompute_layers\", [])}')
            print(f'   No-recompute layers: {strategy.get(\"layerwise_no_recompute_layers\", [])}')
    else:
        print(f'   Recompute: Disabled (best performance)')
    
except Exception as e:
    print(f'   Error reading results: {e}')
"
        fi
        
        echo ""
        echo "🔧 Next Steps:"
        echo "   1. Run training with optimal strategy:"
        echo "      bash scripts/train_with_cost.sh"
        echo ""
        echo "   2. Or run with specific GPUs:"
        echo "      GPUS=$GPUS bash scripts/train_with_cost.sh"
        
    else
        echo "⚠️  Search completed but no optimal solution file found."
        echo "    Check the search logs for details."
    fi
    
else
    echo "❌ Optimized recompute search failed (exit code: $SEARCH_EXIT_CODE)"
    echo "   Please check the error messages above."
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Check if all profile files are valid JSON"
    echo "   2. Verify memory limits are reasonable"
    echo "   3. Try with smaller batch size range"
    echo "   4. Check PaddleNLP installation"
    exit $SEARCH_EXIT_CODE
fi

echo ""
echo "🎉 Optimized fine-grained recompute search completed!" 