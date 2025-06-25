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


# Max_ReLu
# Total graphs: 19424
# Average node degree:        2.6027
# Average nodes per graph:    281.63
# Average edges per graph:    733.01
# Disk (du -h): 8.5G

# Full_NoTemp (train==5240)
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
# Disk (du -h):