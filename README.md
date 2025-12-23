# Context-Aware Attention-Based Graph Representations for Document Classification and Summarization

It is the extension of the previous work on [Attention Document-Graph](https://github.com/buguemar/AttnGraphs/), 
supervised by the original authors, as an individual master's project at Hasso Plattner Institute (HPI) and University of Potsdam.

### 📝 Abstract
This work extends the recent efforts by [Bugueño and de Melo (2025)](https://arxiv.org/abs/2508.00864) in graph-based document modeling by introducing a simple, data-driven approach to capture local and mid-range sentence relations. Using dynamic sliding-window attention, we build graphs that retain key semantic and structural dependencies. Then, we train GAT models on these graphs for two NLP tasks, i.e., document classification and extractive summarization, showing that automatic graph construction methods can yield strong performance with minimal custom-tailoring and economical resources.

---

### 📁 Repository Structure

This repository follows the structure of the original repository with additional files and folders related to the extractive summarization task.

```bash
├── config/
│   ├── Classifier/                         # Config files including parameters for training our attention-based classifier model
│   └── Summerizer/                         # Config files including parameters for training our attention-based summarizer model
├── GNN_Results/                            # Results obtained from our runs
│   ├── Classifier/                         # Results from learned graphs on document classification task
│   └── Summerizer/                         # Results from learned graphs on document extractive summarization task
├── imgs/                                   # Figures shown in the paper and samples of adjacency matrices from each dataset
│   └── analyze_summaries/                  # Figures generated from the summarization analysis
├── src/                                    # Source code for training and evaluation
│   ├── data/                               # Data loaders and utils
│   ├── graphs/                             # Graph-based architectures
│   ├── models/                             # Core training models
│   └── pipeline/                           # Connector for text-graph models  
├── analyze_summaries.py                    # Analysis: analyze GAT prediction on the test split, i.e., t-SNE, Rouge scores, BERTScore and sentence distribution
├── get_structural_graph_stats.py           # Analysis: get information on graph dataset, i.e., number of nodes, edges, node degree and disk size
├── paper.pdf                               # Project report
├── README.md                            
├── requirements.txt                        # Python dependencies
├── train_MHAClassifier.py                  # Training: sliding-window MHA-based classifier training script
├── train_MHASummarizer.py                  # Training: sliding-window MHA-based summarizer training script
├── train_attention_GNN.py                  # Training: GNN training script for document classification
├── train_attention_gnn_node.py             # Training: GNN training script  for document summarization
└── visualize_attns.py                      # Analysis: visualize attention weight matrices from MHA models
```

### 📊 Datasets
The preprocessed versions of Hyperpartisan News Detection (HND) and BBC News datasets are available 
in [the `data/` folder in the original repository](https://github.com/Buguemar/AttnGraphs/tree/main/data).

Alternatively, all three preprocessed datasets can be downloaded from:
[here](https://drive.google.com/drive/folders/1HnI_9cjS9O1b-OaU_2wIRLWvDvglYlbb?usp=sharing).

---

### 🚀 Running the Code

#### Document Classification
To train MHA classifier:
```
python train_MHAClassifier.py -s config/Classifier/your_MHAclassifier_config_file.yaml
```
To train GAT classifier:
```
python train_attention_GNN.py -s config/Classifier/your_GATclassifier_config_file.yaml
```

#### Document Summarization
To train MHA summarizer:
```
python train_MHASummarizer.py -s config/Summarizer/your_MHAsummarizer_config_file.yaml
```
To train GAT summarizer:
```
python train_attention_gnn_node.py -s config/Summarizer/your_GATsummarizer_config_file.yaml
```

#### Analysis

To visualize attentions adjacency matrix,
*required: a trained MHA model, either a classifier or a summarizer*.

`num_print` defines number of samples prints from each split.
`random_sampling` enables random samples instead of the first `<num_print>` instances. 
```
python visualize_attns.py -s config/<folder>/<your_MHA_config_file>.yaml [optional --num_print <int>] [optional --random_sampling]
```

To print structural statistics from a graph dataset,
*required: a graph dataset*.
```
python get_structural_graph_stats.py --data_dir '/path/to/folder/<model_name>/<type_graph>/processed
```

To conduct analyses of your choice on the test split from the best model (with highest Val-f1 score) defined in the GAT config file,
*required: a graph dataset and a trained GNN summarizer*.

`num_print` defines number of samples prints from each split.
`random_sampling` enables random samples instead of the first `<num_print>` instances for t-SNE analysis.

The following analyses is available: `tsne`, `rouge_score`, `bert_score` and `sent_dist`.
```
python analyze_summaries.py -s config/<folder>/<your_GAT_config_file>.yaml [optional --num_print <int>] [optional --random_sampling] [optional --tsne] [optional --rouge_score] [optional --bert_score] [optional --sent_dist]
```
