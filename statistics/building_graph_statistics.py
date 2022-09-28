# Building the graph in the best way possible can strongly influential
# the results I will get

# In this file I want to analise some of that statistics of a certain graph for some parameters
# these statistics include:

# the title of the table: the method used to create graph + hyper-parameter value
# edges_number, num_discontected_nodes, num_components, homophily score for all label_classes, average edges_per_node
import torch

from misc.dataset import DatasetWhole
from misc.helpers import normalizeRNA
from utils import build_graph_radius, build_graph_neighbours
from torch_geometric.utils import homophily

# param ER, PAM, ...
dataset = DatasetWhole('W')

dataset1 = dataset.train['clin']
dataset2 = normalizeRNA(dataset.train['rnanp'])
dataset3 = dataset.train['cnanp']

'''
f = open("demofile2.txt", "a")
f.write("Now the file has more content!")
f.close()
'''

dataset12 = torch.cat([torch.from_numpy(dataset1), torch.from_numpy(dataset2)], dim=1)
dataset23 = torch.cat([torch.from_numpy(dataset2), torch.from_numpy(dataset3)], dim=1)
dataset31 = torch.cat([torch.from_numpy(dataset3), torch.from_numpy(dataset1)], dim=1)

r1 = 7.74  # Clin+RNA
r2 = 8.48  # RNA + CNA
r3 = 9.19  # CNA + CLIN
r = 0.5
while r < 1.5:
    graph1 = build_graph_radius(dataset12, r1*r)
    graph2 = build_graph_radius(dataset23, r2*r)
    graph3 = build_graph_radius(dataset31, r3*r)

    graphs = {graph1: 'clin + rna', graph2: 'rna + cna', graph3: 'cna + clin'}
    for graph in graphs.keys():
        f = open('graph_statistics_R/' + graphs[graph] + '.txt', 'a')

        n_m = graph.num_edges
        mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask[graph.edge_index.view(-1)] = 1
        isolated_nodes = mask[mask == 0].size()[0]
        homos = []

        for s in ['ernp', 'drnp', 'icnp', 'pam50np']:
            Y = torch.from_numpy(dataset.train[s])
            homo = homophily(graph.edge_index, Y)
            homos.append(homo)
        line = '\n' + format(r) + ',' + format(n_m) + ',' + format(isolated_nodes) + ',' + format(
            homos[0]) + ',' + format(homos[1]) + ',' + format(homos[2]) + ',' + format(homos[3])
        f.write(line)

        f.close()
    r += 0.05