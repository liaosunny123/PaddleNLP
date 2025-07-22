from paddlenlp.experimental.galvatron.utils import get_current_all_args
from paddlenlp.experimental.galvatron.search_engine.search_engine import SearchEngine

if __name__ == "__main__":
    print("🚀 Starting Layerwise Recompute Strategy Search...")
    args_dict = get_current_all_args()
    
    # Initialize search engine with layerwise support
    search_engine = SearchEngine(args_dict)
    
    # Run optimization
    results, optimal_solution = search_engine.parallelism_optimization()
    
    print("\n✅ Layerwise Recompute Search Completed!")
    if optimal_solution:
        strategy = optimal_solution['strategy']
        if hasattr(strategy, 'get_recompute_efficiency_info'):
            recompute_info = strategy.get_recompute_efficiency_info()
            print(f"🎯 Best Strategy: {strategy.serialize()}")
            print(f"📊 Recompute: {recompute_info['recompute_layers_count']}/{recompute_info['total_layers']} layers")
        else:
            print(f"🎯 Best Strategy: {strategy.serialize()}")
        print(f"🚀 Throughput: {optimal_solution['throughput']:.2f} samples/s")
        print(f"💾 Memory: {optimal_solution['memory_cost'][0]/1024:.1f}GB")
    else:
        print("❌ No valid solution found. Try adjusting memory limits or batch size range.") 