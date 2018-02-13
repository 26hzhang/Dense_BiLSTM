from utils import load_json
from dataset.prepro import load_vocab
import os


def main():
    data_folder = os.path.join('.', 'dataset', 'data')
    # set tasks
    source_dir = os.path.join(data_folder, 'sst1')
    # load datasets
    trainset = load_json(os.path.join(source_dir, 'train.json'))
    devset = load_json(os.path.join(source_dir, 'dev.json'))
    testset = load_json(os.path.join(source_dir, 'test.json'))
    # load vocabularies
    word_idx, idx_word = load_vocab(os.path.join(source_dir, 'words.vocab'))
    char_idx, idx_chat = load_vocab(os.path.join(source_dir, 'chars.vocab'))
    # TODO
