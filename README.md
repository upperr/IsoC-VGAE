# Isomorphic-Consistent Variational Graph Auto-Encoders for Multi-Level Graph Representation Learning

This is the source code of our work "Isomorphic-Consistent Variational Graph Auto-Encoders for Multi-Level Graph Representation Learning".

## Requirements

- torch 1.11.0

- torch-geometric 2.0.3

## Examples

### node classification

```
python train_node.py --dataset Cora

python train_node.py --dataset CiteSeer

python train_node.py --dataset PubMed
```

### link prediction

```
python train_link.py --dataset Cora

python train_link.py --dataset CiteSeer

python train_link.py --dataset PubMed --bn PairNorm
```

### graph classification

```
python train_graph.py --dataset IMDB-BINARY --seed_data 10

python train_graph.py --dataset COLLAB --seed_data 9
```
