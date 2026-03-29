import os
import random
import math
import torch
import dgl
from dgl import DGLGraph
from torch.utils.data import Dataset
from data.util1 import read_dgl_from_metis


def generate_er_graph(n, p, min_weight, max_weight):
    G = dgl.graph(([], []), num_nodes=n)
    w = -1
    lp = math.log(1.0 - p)
    v = 1
    edges_list = []
    weights = []

    while v < n:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            edges_list.extend([(v, w), (w, v)])
            weight = random.randint(min_weight, max_weight)
            weights.extend([weight, weight])

    if edges_list:
        src, dst = zip(*edges_list)
        G.add_edges(src, dst)
        G.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
    return G


class GraphDataset(Dataset):
    def __init__(
        self, 
        data_dir = None, 
        generate_fn = None,
        ):            
        self.data_dir = data_dir
        self.generate_fn = generate_fn
        if data_dir is not None:
            self.num_graphs = len([
                name 
                for name in os.listdir(data_dir)
                if name.endswith('.METIS')
                ])
        elif generate_fn is not None:
            self.num_graphs = 5000 # sufficiently large number for moving average
        else:
            assert False

    def __getitem__(self, idx):
        if self.generate_fn is None:
            g_path = os.path.join(
                self.data_dir, 
                "{:06d}.METIS".format(idx)
                )
            g = read_dgl_from_metis(g_path)
        else:
            g = self.generate_fn()

        # if 'weight' in g.edata:
        #     print(f"Edge weights for graph {idx}: {g.edata['weight']}")
        # else:
        #     print(f"Graph {idx} has no edge weights.")

        return g
    
    def __len__(self):
        return self.num_graphs

def get_er_15_20_dataset(mode, data_dir = None):
    if mode == "train":
        def generate_fn():
            num_nodes = random.randint(15, 20)
            g = generate_er_graph(num_nodes, 0.15 ,1 ,10)
            return g
        
        return GraphDataset(generate_fn = generate_fn)
    else:    
        return GraphDataset(data_dir = data_dir)    
        
        

