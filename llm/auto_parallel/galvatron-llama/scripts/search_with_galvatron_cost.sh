#!/bin/bash

# Galvatron Cost-Based Layerwise Recompute Search
# 基于 Galvatron 成本模型的细颗粒度重计算策略搜索

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
echo "🚀 Starting Galvatron Cost-Based Layerwise Recompute Search..."

# 搜索参数配置
LAYERNUM=${LAYERNUM:-32}
TOTAL_LAYERS=${TOTAL_LAYERS:-$LAYERNUM}

# GPU配置 - 支持指定GPU列表，确保world_size与实际GPU数量匹配
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | wc -l)
WORLD_SIZE=$GPU_COUNT

# 搜索配置
ENABLE_LAYERWISE_RECOMPUTE_SEARCH=${ENABLE_LAYERWISE_RECOMPUTE_SEARCH:-true}
ENABLE_MEMORY_EFFICIENCY_MODE=${ENABLE_MEMORY_EFFICIENCY_MODE:-true}
SEARCH_GRANULARITY=${SEARCH_GRANULARITY:-"layerwise"}

echo "🔧 Search Configuration:"
echo "   Model layers: $LAYERNUM"
echo "   Total layers for search: $TOTAL_LAYERS"
echo "   GPUs: $GPUS (count: $GPU_COUNT)"
echo "   World size: $WORLD_SIZE"
echo "   Search granularity: $SEARCH_GRANULARITY"
echo "   Enable layerwise search: $ENABLE_LAYERWISE_RECOMPUTE_SEARCH"
echo "   Memory efficiency mode: $ENABLE_MEMORY_EFFICIENCY_MODE"
echo ""

# 批次大小配置
MIN_BSZ=${MIN_BSZ:-2}
MAX_BSZ=${MAX_BSZ:-4}
BSZ_STEP=${BSZ_STEP:-1}
MEMORY_LIMIT=${MEMORY_LIMIT:-16}

echo "   Batch size range: $MIN_BSZ - $MAX_BSZ (step: $BSZ_STEP)"
echo "   Memory limit: ${MEMORY_LIMIT}GB"

# 根据GPU数量调整并行度上限
MAX_TP_SIZE=${MAX_TP_SIZE:-$WORLD_SIZE}
MAX_PP_SIZE=${MAX_PP_SIZE:-$WORLD_SIZE}

echo "   Max TP size: $MAX_TP_SIZE"
echo "   Max PP size: $MAX_PP_SIZE"

# 构建搜索配置参数
SEARCH_CONFIG="
--model_name llama_layerwise_cost
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
--search_granularity $SEARCH_GRANULARITY
--total_layers $TOTAL_LAYERS
--enable_layerwise_recompute_search $ENABLE_LAYERWISE_RECOMPUTE_SEARCH
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
echo "🔬 Running Galvatron Cost-Based Layerwise Search..."
echo "   Using GPUs: $GPUS"
export CUDA_VISIBLE_DEVICES="$GPUS"

# 设置Python路径
PADDLENLP_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
export PYTHONPATH="$PADDLENLP_ROOT:$PYTHONPATH"
echo "   PaddleNLP path: $PADDLENLP_ROOT"

# 运行 layerwise 搜索引擎
python search_layerwise_dist.py $SEARCH_CONFIG

SEARCH_EXIT_CODE=$?

echo ""
if [ $SEARCH_EXIT_CODE -eq 0 ]; then
    echo "✅ Galvatron Cost-Based Layerwise Search completed successfully!"
    
    # 检查结果
    if [ -f "./configs/optimal_solution.json" ]; then
        echo ""
        echo "📊 Search Results:"
        echo "   Optimal solution saved to: ./configs/optimal_solution.json"
        
        # 详细的结果显示
        if command -v python3 >/dev/null 2>&1; then
            echo ""
            echo "📋 Optimal Layerwise Strategy Summary:"
            python3 -c "
import json
try:
    with open('./configs/optimal_solution.json', 'r') as f:
        data = json.load(f)
    
    strategy = data.get('strategy', {})
    performance = data.get('performance_metrics', {})
    recompute_summary = data.get('optimal_recompute_summary', {})
    search_metadata = data.get('search_metadata', {})
    
    print(f'   🎯 Optimal Strategy: {strategy.get(\"strategy_name\", \"Unknown\")}')
    print(f'   ⚙️  Parallelism: PP={strategy.get(\"pp_size\", 1)}, TP={strategy.get(\"tp_size\", 1)}, DP={strategy.get(\"dp_size\", 1)}')
    print(f'   🔄 Sharding: Stage {strategy.get(\"sharding_stage\", 0)}')
    print(f'   📦 Batch Size: {data.get(\"batch_size\", \"Unknown\")}')
    print(f'   🔢 Accumulation Steps: {data.get(\"accumulation_steps\", \"Unknown\")}')
    print(f'   🚀 Throughput: {performance.get(\"throughput_samples_per_sec\", 0):.2f} samples/s')
    print(f'   💾 Memory Usage: {performance.get(\"memory_usage_gb\", [0])[0]:.1f}GB')
    print()
    
    # 详细的重计算信息
    if recompute_summary.get('enabled', False):
        recompute_count = recompute_summary.get('recompute_layers_count', 0)
        total_layers = recompute_summary.get('total_layers', 0)
        recompute_ratio = recompute_summary.get('recompute_ratio', '0%')
        
        print(f'   🧠 Layerwise Recompute Configuration:')
        print(f'      ├─ Enabled layers: {recompute_count}/{total_layers} ({recompute_ratio})')
        print(f'      ├─ Strategy: {recompute_summary.get(\"efficiency_explanation\", \"Unknown\")}')
        
        # 显示具体的 layerwise pattern
        layerwise_pattern = strategy.get('layerwise_recompute', [])
        if layerwise_pattern:
            pattern_str = ''.join(['R' if x == 1 else 'N' for x in layerwise_pattern])
            print(f'      ├─ Pattern: {pattern_str} (R=Recompute, N=Normal)')
            
            recompute_layers = strategy.get('layerwise_recompute_layers', [])
            no_recompute_layers = strategy.get('layerwise_no_recompute_layers', [])
            
            if recompute_layers:
                recompute_layers_str = ', '.join(map(str, recompute_layers))
                print(f'      ├─ Recompute layers: [{recompute_layers_str}]')
            
            if no_recompute_layers:
                no_recompute_layers_str = ', '.join(map(str, no_recompute_layers))
                print(f'      └─ Normal layers: [{no_recompute_layers_str}]')
        
        print()
        print(f'   📈 Performance Trade-off:')
        print(f'      ├─ Memory efficiency: {(recompute_count / total_layers * 100):.1f}% layers use recompute')
        print(f'      └─ Performance impact: Calculated using precise Galvatron cost models')
    else:
        print(f'   🧠 Recompute: Disabled (maximum performance mode)')
        print(f'      └─ All {recompute_summary.get(\"total_layers\", 0)} layers use normal computation')
    
    print()
    print(f'   🔍 Search Statistics:')
    print(f'      ├─ Search space: {search_metadata.get(\"search_space_size\", 0)} combinations')
    print(f'      ├─ Evaluated: {search_metadata.get(\"evaluated_combinations\", 0)} combinations')
    print(f'      ├─ Memory limit: {search_metadata.get(\"memory_upper_limit_gb\", 0)}GB')
    print(f'      └─ World size: {search_metadata.get(\"world_size\", 0)} GPUs')
    
except Exception as e:
    print(f'   ❌ Error reading results: {e}')
"
        fi
        
        echo ""
        echo "🔧 Next Steps:"
        echo "   1. Use the optimal strategy for training:"
        echo "      bash scripts/train_with_layerwise_cost.sh"
        echo ""
        echo "   2. Run with specific configuration:"
        echo "      GPUS=$GPUS LAYERNUM=$LAYERNUM bash scripts/train_with_layerwise_cost.sh"
        echo ""
        echo "   3. Customize search parameters:"
        echo "      TOTAL_LAYERS=16 MEMORY_LIMIT=24 bash scripts/search_with_galvatron_cost.sh"
        
    else
        echo "⚠️  Search completed but no optimal solution file found."
        echo "    Check the search logs for details."
    fi
    
else
    echo "❌ Galvatron Cost-Based Layerwise Search failed (exit code: $SEARCH_EXIT_CODE)"
    echo "   Please check the error messages above."
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Verify all profile files are valid JSON"
    echo "   2. Check memory limits are reasonable for your hardware"
    echo "   3. Try reducing batch size range or layer count"
    echo "   4. Ensure PaddleNLP installation is complete"
    echo "   5. Check GPU availability and CUDA setup"
    exit $SEARCH_EXIT_CODE
fi