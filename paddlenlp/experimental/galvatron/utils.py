import json
import os
from typing import List
import sys
from dataclasses import dataclass, field

def read_json_config(path):
    if os.path.exists(path) == False:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fp:
            fp.write("{}")
    return json.load(open(path, 'r', encoding="utf-8"))

def write_json_config(path, config):
    with open(path, 'w') as fp:
        json.dump(config, fp, indent=4)

def num2str(num, type:str):  # NOTE this function maybe have some bugs
    if type == 'seq':
        if isinstance(num, List) and len(num) == 1:
            num = num[0]        
        if isinstance(num, List):
            info = f'seq[{",".join(map(str, num))}]'
        else:
            info = f'seq{num}'
    elif type == 'layernum':
        info = 'layernum' + '[' + str(num) + ']'
    return info

@dataclass
class Strategy:
    pp_size: int = field(default=1, metadata={"help": "The number of processes to use for parallel processing."})
    tp_size: int = field(default=1, metadata={"help": "The number of threads to use for parallel processing."})
    dp_size: int = field(default=1, metadata={"help": "The number of data parallelism to use."})
    sharding_stage: int = field(default=0, metadata={"help": "The stage of sharding. 0: no sharding, 1: sharding1, 2: sharding2, 3: sharding3"})
    recompute: int = field(default=0, metadata={"help": "Whether to use recompute."})
    layerwise_recompute: List[int] = field(default_factory=list, metadata={"help": "Layerwise recompute configuration. List of 0/1 for each layer."})
    
    def serialize(self):
        text = f'pp{self.pp_size}_tp{self.tp_size}_dp{self.dp_size}_stage{self.sharding_stage}_recompute{self.recompute}'
        if self.layerwise_recompute:
            layerwise_str = ''.join(map(str, self.layerwise_recompute))
            text += f'_layerwise{layerwise_str}'
        return text
    
    def deserialize(self, text):
        if isinstance(text, str):
            items = text.split('_')
            for item in items:
                if 'pp' in item:
                    self.pp_size = int(item.split('pp')[1])
                elif 'tp' in item:
                    self.tp_size = int(item.split('tp')[1])
                elif 'dp' in item:
                    self.dp_size = int(item.split('dp')[1])
                elif 'stage' in item:
                    self.sharding_stage = int(item.split('stage')[1])
                elif 'recompute' in item and not 'layerwise' in item:
                    self.recompute = int(item.split('recompute')[1])
                elif 'layerwise' in item:
                    layerwise_str = item.split('layerwise')[1]
                    self.layerwise_recompute = [int(x) for x in layerwise_str]
        elif isinstance(text, dict):
            self.pp_size = text.get('pp_size', self.pp_size)
            self.tp_size = text.get('tp_size', self.tp_size)
            self.dp_size = text.get('dp_size', self.dp_size)
            self.sharding_stage = text.get('sharding_stage', self.sharding_stage)
            self.recompute = text.get('recompute', self.recompute)
            self.layerwise_recompute = text.get('layerwise_recompute', self.layerwise_recompute)
        elif isinstance(text, List):
            if len(text) >= 5:
                self.pp_size = text[0]
                self.tp_size = text[1]
                self.dp_size = text[2]
                self.sharding_stage = text[3]
                self.recompute = text[4]
                if len(text) > 5:
                    self.layerwise_recompute = text[5] if isinstance(text[5], list) else []
        else:
            raise ValueError("Unsupported type for deserialization. Supported types are str, dict, and list.")
    
    def get_layerwise_recompute_count(self):
        """Get the number of layers with recompute enabled"""
        return sum(self.layerwise_recompute) if self.layerwise_recompute else 0
    
    def set_layerwise_recompute(self, total_layers, recompute_count):
        """Set layerwise recompute pattern: first recompute_count layers enabled, rest disabled"""
        if recompute_count > total_layers:
            recompute_count = total_layers
        self.layerwise_recompute = [1] * recompute_count + [0] * (total_layers - recompute_count)
        # Update the global recompute flag based on whether any layer has recompute
        self.recompute = 1 if recompute_count > 0 else 0
    
    def get_recompute_efficiency_info(self):
        """Get efficiency information about the recompute configuration"""
        if not self.layerwise_recompute:
            return {"enabled": False, "total_layers": 0, "recompute_layers_count": 0, "recompute_ratio": "0%"}
        
        total_layers = len(self.layerwise_recompute)
        recompute_count = self.get_layerwise_recompute_count()
        recompute_ratio = f"{(recompute_count / total_layers * 100):.1f}%" if total_layers > 0 else "0%"
        
        # Generate efficiency explanation
        if recompute_count == 0:
            explanation = "No recompute - best performance, highest memory usage"
        elif recompute_count == total_layers:
            explanation = "Full recompute - lowest memory usage, slower performance"
        else:
            explanation = f"Selective recompute - balanced memory-performance trade-off"
            
        return {
            "enabled": recompute_count > 0,
            "total_layers": total_layers,
            "recompute_layers_count": recompute_count,
            "recompute_ratio": recompute_ratio,
            "efficiency_explanation": explanation,
            "layerwise_recompute_layers": [i for i, x in enumerate(self.layerwise_recompute) if x == 1],
            "layerwise_no_recompute_layers": [i for i, x in enumerate(self.layerwise_recompute) if x == 0]
        }
    
    def __str__(self):
        return self.serialize()
    
def get_current_all_args():
    args_dict = {}
    i = 0
    argv = sys.argv
    while i < len(argv):
        arg = argv[i]
        if arg.startswith('-'):
            if i + 1 < len(argv) and not argv[i + 1].startswith('-'):
                args_dict[arg] = argv[i + 1]
                i += 2  
            else:
                args_dict[arg] = True
                i += 1
        else:
            i += 1
    return args_dict