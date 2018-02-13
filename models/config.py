import os
from dataset import load_vocab
from utils import load_json, load_embeddings


class Config(object):
    def __init__(self, source_dir):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.word_vocab = load_vocab(os.path.join(source_dir, 'words.vocab'))
        self.char_vocab = load_vocab(os.path.join(source_dir, 'chars.vocab'))
        self.vocab_size = len(self.word_vocab)
        self.char_vocab_size = len(self.char_vocab)
        self.label_size = load_json(os.path.join(source_dir, 'label.json'))["label_size"]
        self.word_emb = load_embeddings(os.path.join(source_dir, 'glove.filtered.npz'))

    # log and model file paths
    ckpt_path = './ckpt/'  # checkpoint path
    max_to_keep = 5  # max model to keep while training
    no_imprv_patience = 5

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
    l2_reg = 0.0
    grad_clip = 5.0
    lr = 0.005
    lr_decay = 0.05
    keep_prob = 0.5
