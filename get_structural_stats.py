import torch
from torch_geometric.utils import degree
from tqdm import tqdm
import os

folder = '/scratch2/rldallitsako/datasets/AttnGraphs_GovReports/Extended_ReLu/max_unified/processed'
total_degree = 0
total_nodes = 0
total_edges = 0
num_graphs = 0

# List all matching data files first
files = [f for f in os.listdir(folder) if f.startswith('data') and f.endswith('.pt')]

for filename in tqdm(files, desc="Processing graphs"):
    if filename.startswith('data') and filename.endswith('.pt'):
        path = os.path.join(folder, filename)
        data = torch.load(path, weights_only=False)

        num_graphs += 1
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1)

        deg = degree(data.edge_index[0], num_nodes=num_nodes)

        total_degree += deg.sum().item()
        total_nodes += num_nodes
        total_edges += num_edges

# Final stats
avg_degree = total_degree / total_nodes if total_nodes > 0 else 0
avg_nodes_per_graph = total_nodes / num_graphs if num_graphs > 0 else 0
avg_edges_per_graph = total_edges / num_graphs if num_graphs > 0 else 0

print(f"Total graphs: {num_graphs}")
print(f"Average node degree:        {avg_degree:.4f}")
print(f"Average nodes per graph:    {avg_nodes_per_graph:.2f}")
print(f"Average edges per graph:    {avg_edges_per_graph:.2f}")

################################################################################# GR

# Max_ReLu
# Total graphs: 19424
# Average node degree:        2.6027
# Average nodes per graph:    281.63
# Average edges per graph:    733.01
# Disk (du -h): 8.5G

# Full_NoTemp (train==5270)
# Total graphs: 7212
# Average node degree:        388.8547
# Average nodes per graph:    281.43
# Average edges per graph:    109434.57
# Disk (du -h): 30G

# Mean_Sigmoid
# Total graphs: 19424
# Average node degree:        28.8773
# Average nodes per graph:    281.63
# Average edges per graph:    8132.69
# Disk (du -h): 14G

# Max_Sigmoid
# Total graphs: 19424
# Average node degree:        3.1265
# Average nodes per graph:    281.63
# Average edges per graph:    880.52
# Disk (du -h): 8.6G

# Max_NoTemp (train==5240)
# Total graphs: 7182
# Average node degree:        2.5146
# Average nodes per graph:    281.54
# Average edges per graph:    707.97
# Disk (du -h): no data

# Max_NoTemp (train==999) - 404
# Total graphs: 2942
# Average node degree:        2.5800
# Average nodes per graph:    277.85
# Average edges per graph:    716.84
# Disk (du -h): 1.3G

# full_NoTemp (train==999) - 404
# Total graphs: 2942
# Average node degree:        383.0287
# Average nodes per graph:    280.23
# Average edges per graph:    107335.92
# Disk (du -h): 12G

# Mean_NoTemp (train==999) - 404
# Total graphs: 2942
# Average node degree:        20.2891
# Average nodes per graph:    277.85
# Average edges per graph:    5637.24
# Disk (du -h): 1.8G

################################################################################# HND
# Full_ReLu
# Average node degree:        39.0267
# Average nodes per graph:    19.68
# Average edges per graph:    768.15
# Disk (du -h): 72M

# Mean_ReLu
# Average node degree:        9.3197
# Average nodes per graph:    19.68
# Average edges per graph:    183.44
# Disk (du -h): 47M

# Max_ReLu
# Average node degree:        2.8676
# Average nodes per graph:    19.68
# Average edges per graph:    56.44
# Disk (du -h): 42M

################################################################################# BBC
# Full_NoTemp
# Total graphs: 2167
# Average node degree:        27.8451
# Average nodes per graph:    19.30
# Average edges per graph:    537.42
# Disk (du -h): 113M

# Mean_NoTemp
# Total graphs: 2167
# Average node degree:        7.8778
# Average nodes per graph:    19.30
# Average edges per graph:    152.04
# Disk (du -h): 84M

# Max_NoTemp
# Total graphs: 2167
# Average node degree:        3.4711
# Average nodes per graph:    19.30
# Average edges per graph:    66.99
# Disk (du -h): 77M