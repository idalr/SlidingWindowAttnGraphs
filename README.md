# Attention-DocumentGraphs
Attention-based document graphs for long document downstream tasks: classification and extractive summarization.

### Data 

| Dataset                      | Task                    | Source      | Description | #samples                        |
|------------------------------|-------------------------|-------------|:-----------:|---------------------------------|
| Hyperpartisan News Detection | Document Classification | [Zenodo](https://zenodo.org/records/5776081)  (Although it comprises two parts, this study only uses *byarticle*)     | A collection of 645 news articles labeled according to whether it shows blind or unreasoned allegiance to one party or entity. The dataset exhibits a minor class imbalance.   | 645/625 (train/test)            |
| GovReport                   | Document Summarization  | [HuggingFace](https://huggingface.co/datasets/ccdv/govreport-summarization) (adapted from https://gov-report-data.github.io/)|  Reports written by government research agencies including Congressional Research Service and U.S. Government Accountability Office. Compared with other long document summarization datasets, GovReport dataset has longer summaries and documents and requires reading in more context to cover salient words to be summarized.    | 17,517/973/973 (train/val/test) |


The corresponding datasets have already been preprocessed. Sentence vocabulary and cleaned documents are available [here](https://drive.google.com/drive/folders/1HnI_9cjS9O1b-OaU_2wIRLWvDvglYlbb?usp=sharing).


### Learning Graphs

#### Preliminaries
```config/``` folder with all the instructions and parameters for MHA models and predict-files in the case of summarization tasks. 

```config_gnn/``` folder with all the instructions and parameters for training the GAT models exclusively for attention-based graphs

```config_heuristics_gnn/``` folder with all the instructions and parameters for training the GAT models exclusively for heuristic-based graphs  

#### Document Classification

Training MHAClassifier as a multiclass classification (hyperpartisan or not). The model learns the dependencies between every par of sentences in a document and reduce the learned embeddings to a final linear layer.
```python
python train_MHAClassifier.py -s config/Classifier/your_file.yaml
```

Create graph objects and store them in the speficied folder. Afterwards, a GAT for graph classification is trained. Please note that this python file serves either for learned-graphs (```config_gnn```).

```python
python train_GNN.py -s config_gnn/Classifier/type_of_model/your_file.yaml
```

#### Document Summarization

Training MHASummarizer as a multilabel task. The model learns the dependencies between every par of sentences in a document and every sentence has a label associated.  

```python
python train_MHASummarizer.py  -s config/Summarizer/your_file.yaml
```

Create files in /raw graph-version of the summarization dataset with a filename that is need for the Pytorch-Geometric graphs creation. The corresponding graphs are stored in the specified folder.
Train a GAT for node classification. This python file serves either for learned-graphs (```config_gnn```).  
```python
python create_graphSum_files.py -s config_gnn/Summarizer/type_of_model/your_file.yaml
python train_gnn_node.py -s config_gnn/Summarizer/type_of_model/your_file.yaml
```
