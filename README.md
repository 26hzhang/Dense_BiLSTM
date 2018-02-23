# Densely Connected Bidirectional LSTM
![Authour](https://img.shields.io/badge/Author-Zhang%20Hao%20(Isaac%20Changhau)-blue.svg) ![](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![](https://img.shields.io/badge/TensorFlow-1.4.0-yellowgreen.svg)  
Tensorflow Implementation of **Densely Connected Bidirectional LSTM with Applications to Sentence Classification**, [[arXiv:1802.00889]](https://arxiv.org/pdf/1802.00889.pdf).

### Densely Connected Bidirectional LSTM (DC-Bi-LSTM) Overview
![model_graph_1](/docs/model_graph_1.png)
> The architecture of DC-Bi-LSTM. The first-layer reading memory is obtained based on original input sequence, and second-layer reading memory based on the position-aligned concatenation of original input sequence and first-layer reading memory, and so on. Finally, get the n-th-layer reading memory and take it as the final feature representation for classification.

![model_graph_2](/docs/model_graph_2.png)
> Illustration of (a) Deep Stacked Bi-LSTM and (b) DC-Bi-LSTM. Each black node denotes an input layer. Purple, green, and yellow nodes denote hidden layers. Orange nodes denote average pooling of forward or backward hidden layers. Each red node denotes a class. Ellipse represents the concatenation of its internal nodes. Solid lines denote the connections of two layers. Finally, dotted lines indicate the operation of copying.

### Dataset Overview
*More details of datasets are shown here*: [[dataset/raw/README.md]](/dataset/raw)

**Data** | Classes | Average sentence length | Dataset size | Vocab size | Number of words present in word2vec | Test size
:---: | :---: | :---: | :---: | :---: | :---: | :---:
MR | 2 | 20 | 10662 | 18765 | 16448 | CV
SST1 | 5 | 18 | 11855 | 17836 | 16262 | 2210
SST2 | 2 | 19 | 9613 | 16185 | 14838 | 1821
Subj | 2 | 23 | 10000 | 21323 | 17913 | CV
TREC | 6 | 10 | 5952 | 9592 | 9125 | 500
CR | 2 | 19 | 3775 | 5340 | 5046 | CV
MPQA | 2 | 3 | 10606 | 6246 | 6083 | CV

### Usage
**Configuration**: all parameters and configurations are maintained in [models/config.py](/models/config.py).  
The first step is to prepare the required data (pre-trained word embeddings and raw datasets). The raw datasets are already included in this repository, which are located at `dataset/raw/`, word embeddings used in the paper, the _300-dimensional Glove vectors that were trained on 42 billion words_, can be obtained by
```bash
$ cd dataset
$ ./download_emb.sh
```
After downloading the pre-trained word embeddings, run following to build training, development and testing dataset among all raw datasets, the built datasets will be stored in `dataset/data/` directory.
```bash
$ cd dataset
$ python3 prepro.py
```
Then training model on a specific dataset via
```bash
$ python3 train_model.py <dataset_folder_name>
# eg:
$ python3 train_model.py subj
```
If everything goes properly, then training process will be launched
```bash
...
No checkpoint found in directory ./ckpt/subj, cannot resume training. Do you want to start a new training session?
(y)es | (n)o: y
Start training...
Epoch  1/30:
40/40 [==============================] - 476s - train loss: 0.5247     
Testing model over DEVELOPMENT dataset: accuracy - 89.700
 -- new BEST score on DEVELOPMENT dataset: 89.700
Testing model over TEST dataset: accuracy - 90.500
Epoch  2/30:
40/40 [==============================] - 485s - train loss: 0.3025     
Testing model over DEVELOPMENT dataset: accuracy - 91.200
 -- new BEST score on DEVELOPMENT dataset: 91.200
Testing model over TEST dataset: accuracy - 91.800
Epoch  3/30:
40/40 [==============================] - 482s - train loss: 0.2538     
Testing model over DEVELOPMENT dataset: accuracy - 92.200
 -- new BEST score on DEVELOPMENT dataset: 92.200
Testing model over TEST dataset: accuracy - 92.900
...
```

### Demo Results
Here I only test run the model on several datasets with several epoch to validate if this model works properly.

**Data** | Dev | Test | Epochs
:---: | :---: | :---: | :---:
MR | _TODO_ | _TODO_ | N.A.
SST1 | _TODO_ | _TODO_ | N.A.
SST2 | _TODO_ | _TODO_ | N.A.
Subj | 93.1 | 93.8 | 5
TREC | _TODO_ | _TODO_ | N.A.
CR | _TODO_ | _TODO_ | N.A.
MPQA | _TODO_ | _TODO_ | N.A.
