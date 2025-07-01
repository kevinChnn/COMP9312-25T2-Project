################################################################################
# Do not edit this code cell.
from itertools import combinations, accumulate
################################################################################

class UndirectedUnweightedGraph:
    def __init__(self, edge_list):
        info = True
        for u, v in edge_list:
            if info:
                info = False
                self.vertex_num, self.edge_num = u, v
                self.adj_list = [list() for _ in range(self.vertex_num)]
            else:
                self.adj_list[u].append(v)
                self.adj_list[v].append(u)
                
# The following code is used to generate random graphs. You don't need to read this.
    class _LCG:
        _a, _c, _m = 6364136223846793005, 1442695040888963407, 2 ** 64
        __slots__ = ("state",)

        def __init__(self, seed=42):
            self.state = seed & 0xFFFFFFFFFFFFFFFF

        def rand64(self):
            self.state = (self._a * self.state + self._c) % self._m
            return self.state

        def randint(self, lo, hi):
            return lo + self.rand64() % (hi - lo + 1)

        def sample(self, population, k):
            pop = list(population)
            n = len(pop)
            for i in range(k):
                j = i + self.rand64() % (n - i)
                pop[i], pop[j] = pop[j], pop[i]
            return pop[:k]

    @staticmethod
    def generate_dense_diverse_graph(num_cliques: int,
                                     size_range=(5, 10),
                                     bridge_factor=3,   
                                     inter_p=0.15,       
                                     seed: int = 42):
        rng = UndirectedUnweightedGraph._LCG(seed)
        min_c, max_c = size_range

        clique_sizes = [rng.randint(min_c, max_c) for _ in range(num_cliques)]
        prefix = [0] + list(accumulate(clique_sizes)) 
        edges = []

        for idx, sz in enumerate(clique_sizes):
            base = prefix[idx]
            for i, j in combinations(range(sz), 2):
                edges.append([base + i, base + j])

        for i in range(num_cliques):
            for j in range(i + 1, num_cliques):
                trials = rng.randint(0, 3)   
                for _ in range(trials):
                    if rng.rand64() / 2**64 < inter_p:
                        u = prefix[i] + rng.randint(0, clique_sizes[i]-1)
                        v = prefix[j] + rng.randint(0, clique_sizes[j]-1)
                        edges.append([u, v])

        num_bridges = bridge_factor * num_cliques
        next_id = prefix[-1]

        for _ in range(num_bridges):
            s = rng.randint(2, num_cliques)           
            chosen = rng.sample(range(num_cliques), s)   
            bridge = next_id
            next_id += 1
            for c_idx in chosen:
                base = prefix[c_idx]
                for v in range(clique_sizes[c_idx]):
                    edges.append([bridge, base + v])
        vertex_num = next_id
        edge_num = len(edges)
        edge_list = [[vertex_num, edge_num]] + edges
        k_auto = min(clique_sizes) - 1 

        return edge_list
    
    
import time
from time import time
import numpy as np

if __name__ == "__main__":

    print('\n######## Loading the dataset...')
    test = [
        (15, (9, [51, 82, 7, 22, 6]), (6, [11, 140, 11, 6])),
        (20, (7, [29, 169, 15, 7]), (6, [10, 183, 19, 5, 3])),
        (80, (9, [108, 688, 33, 16, 9]), (14, [790, 64]))
        ]

    for c in test:
        edge_list = UndirectedUnweightedGraph.generate_dense_diverse_graph(c[0], seed=42) # Use fixed seed for reproducibility
        dataSetFile = f"data_{c[0]}.graph.txt"
        with open(dataSetFile, 'w') as f:
            for edge in edge_list:
                f.write(f"{edge[0]} {edge[1]}\n")
        for k, ans in c[1:]:
            ansFile = f"ans_{c[0]}_{k}.txt"
            ans = " ".join(map(str, ans))
            with open(ansFile, 'w') as f:
                f.write(f"{k} {ans}\n")