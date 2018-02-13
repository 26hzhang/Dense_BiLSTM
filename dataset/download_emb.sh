#!/usr/bin/env bash

# create embedding folder
mkdir embedding
# Download GloVe 42B
DATA_DIR=./embedding
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -O ${DATA_DIR}/glove.42B.zip
unzip ${DATA_DIR}/glove.42B.zip -d ${DATA_DIR}
# remove zip file
rm ${DATA_DIR}/glove.42B.zip
