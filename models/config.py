import numpy as np


class Config(object):
    def __init__(self):
        pass

    ckpt_path = './ckpt/'  # checkpoint path
    max_to_keep = 5  # max model to keep while training

    # dataset
    vocab_size = None
    char_vocab_size = None
    label_size = 2

    # word embeddings
    use_word_emb = True
    finetune_emb = False
    word_dim = 300

    # char embeddings
    use_char_emb = False
    char_dim = 100
    char_rep_dim = 100
    # CNN filter size and height for char representation
    filter_sizes = [50, 50]  # sum of filter sizes should equal to char_out_size
    heights = [5, 5]

    # highway network
    use_highway = False
    highway_num_layers = 2

    # model parameters
    num_layers = 5
    num_units = 20
    num_units_last = 100

    # hyperparameters
    l2_reg = 0.1
    lr = 0.005
    grad_clip = 5.0
    lr_decay = 0.05
    keep_prob = 0.5


def load_embeddings(filename):
    try:
        with np.load(filename) as data:
            return data['embeddings']
    except IOError:
        raise 'ERROR: Unable to locate file {}.'.format(filename)
