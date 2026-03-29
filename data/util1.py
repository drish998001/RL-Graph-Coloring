import torch
import dgl
import random
import json

def read_dgl_from_metis(metis_path, min_weight=1, max_weight=10):
    edges_undirected = set()

    with open(metis_path, "r") as f:
        lines = f.readlines()
        num_nodes, num_edges = list(map(int, lines[0].split()))

        for u, line in enumerate(lines[1:]):
            nums = list(map(int, line.split()))
            for v in nums:
                if u != v:
                    # store undirected edge as sorted pair
                    edges_undirected.add(tuple(sorted((u, v))))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)

    if edges_undirected:
        src = []
        dst = []
        weights = []

        for u, v in edges_undirected:
            # sample a single weight for the undirected edge {u,v}
            w = random.randint(min_weight, max_weight)

            # add both directions with the same weight
            src.extend([u, v])
            dst.extend([v, u])
            weights.extend([w, w])

        g.add_edges(src, dst)
        g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
    else:
        g.edata['weight'] = torch.empty(0, dtype=torch.float32)

    return g

def write_nx_to_metis(g, path): 
    with open(path, "w") as f:
        # write the header
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        f.write("{} {}\n".format(num_nodes, num_edges))
        
        # now write all edges
        for u in g.nodes():
            sorted_adj_nodes = sorted([v for v in g[u]])
            neighbors = " ".join([str(v) for v in sorted_adj_nodes])
            f.write("{}\n".format(neighbors))
