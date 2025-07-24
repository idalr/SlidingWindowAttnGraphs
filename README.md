# Context-Aware Attention-Based Graph Representations for Document Classification and Summarization

It is the extension of the previous work on [Attention Document-Graph](https://github.com/buguemar/AttnGraphs/), 
supervised by the original authors, as an individual master's project at Hasso Plattner Institute (HPI) and University of Potsdam.

### 📝 Abstract
*To be added*

---

### 📁 Repository Structure

This repository follows the structure of the original repository with additional files and folders related to the extractive summarization task.

```bash
├── analyses/                               # Generate figures appeared in the paper
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
├── paper.pdf                                # Project report
├── README.md                            
├── requirements.txt                        # Python dependencies
├── train_MHAClassifier.py                  # Sliding-window MHA-based Classifier Training script
├── train_MHASummarizer.py                  # Sliding-window MHA-based Summarizer Training script
├── train_attention_GNN.py                  # GNN Training script for document classification
└── train_attention_gnn_node.py             # GNN Training script  for document summarization
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
python train_MHAClassifier.py --config config/Classifier/your_MHAclassifier_config_file.yaml
```
To train GAT classifier:
```
python train_attention_GNN.py --config config/Classifier/your_GATclassifier_config_file.yaml
```

#### Document Summarization
To train MHA summarizer:
```
python train_MHASummarizer.py --config config/Summarizer/your_MHAsummarizer_config_file.yaml
```
To train GAT summarizer:
```
python train_attention_gnn_node.py --config config/Summarizer/your_GATsummarizer_config_file.yaml
```

#### Analysis

To visualize attentions adjacency matrix.
`num_print` defines number of samples prints from each split.
`random` enables random samples instead of the first `<num_print>` instances. 
```
python visualize_attns.py --config config/<folder>/<your_MHA_config_file>.yaml [optional --num_print <int>] [optional --random]
```

To print structural statistics from a graph dataset.
```
python get_structural_graph_stats.py --data-dir '/path/to/folder/<model_name>/<type_graph>/processed
```

To conduct analyses of your choice on the test split from the best model (with highest Val-f1 score) defined in the GAT config file.
`num_print` defines number of samples prints from each split.
`random` enables random samples instead of the first `<num_print>` instances for t-SNE.

The following analyses is available: `tsne`, `rouge_score`, `bert_score` and `sent_dist`.
```
python analyze_summaries.py --config config/<folder>/<your_GAT_config_file>.yaml [optional --num_print <int>] [optional --random] [optional --tsne] [optional --rouge_score] [optional --bert_score] [optional --sent_dist]
```