import torch
from torch_geometric.utils import degree
from tqdm import tqdm
import os
import argparse
import subprocess

def main_run(args):

    data_dir = args.data_dir
    # Validate it's a directory
    if not os.path.isdir(data_dir):
        raise ValueError(f"{data_dir} is not a valid directory")

    # List all matching data files
    files = [f for f in os.listdir(data_dir) if f.startswith('data') and f.endswith('.pt')]

    # Initialize lists for counting
    total_degree = 0
    total_nodes = 0
    total_edges = 0
    num_graphs = 0
    for filename in tqdm(files, desc="Processing graphs"):
        if filename.startswith('data') and filename.endswith('.pt'):
            path = os.path.join(data_dir, filename)
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

    dir_size = subprocess.check_output(["du", "-sh", data_dir]).split()[0].decode()
    file_results = 'graph_stat_ana'

    with open(file_results + '.txt', 'a') as f:
        print("================================================")
        print("================================================", file=f)
        print(f"Analyzed folder: {data_dir}", file=f)
        print(f"Graph dir size: {dir_size}")
        print(f"Graph dir size: {dir_size}", file=f)
        print(f"Total graphs: {num_graphs}")
        print(f"Total graphs: {num_graphs}", file=f)
        print(f"Average node degree:        {avg_degree:.4f}")
        print(f"Average node degree:        {avg_degree:.4f}", file=f)
        print(f"Average nodes per graph:    {avg_nodes_per_graph:.2f}")
        print(f"Average nodes per graph:    {avg_nodes_per_graph:.2f}", file=f)
        print(f"Average edges per graph:    {avg_edges_per_graph:.2f}")
        print(f"Average edges per graph:    {avg_edges_per_graph:.2f}", file=f)
        print("================================================", file=f)
        f.close()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="path to the input directory",
    )
    args = arg_parser.parse_args()

    main_run(args)