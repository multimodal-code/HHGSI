# HHGSI

## <1> Introduction 

Code for HHGSI model.

## <2> How to use

```bash
python HHGSI.py [parameters]

# enable GPU
python HHGSI.py --cuda 1
```


We utilize four datasets: CLEF, MIR, PASCAL, NUS-WIDE. You can get the original data <a href="https://snap.stanford.edu/data/web-flickr.html">here</a>.

## <3> Data requirement

het_neigh.txt: generated neighbor set of each node by random walk with re-start 

het_random_walk.txt: generated random walks as node sequences (corpus) for model training

i_i_list.txt: image neighbor list of each image

node_net_embedding.txt: pre-trained node embedding by network embedding
