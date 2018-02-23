import os
import re
import sys
import json
import codecs
import operator
import numpy as np
from tqdm import tqdm

np.random.seed(1234)
special_chars = re.compile(r'[^A-Za-z_\d,.?!;:$\- \'\"]', re.IGNORECASE)

source_dir = os.path.join('.', 'raw')
emb_dir = os.path.join('.', 'embedding')
target_dir = os.path.join('.', 'data')

# special tokens
PAD = '__PAD__'
UNK = '__UNK__'


def clean_text(text):
    """Remove special tokens and clean up text"""
    text = text.replace("``", '"').replace("''", '"').replace("`", "'")  # convert quote symbols
    text = text.replace("n 't", "n't").replace("can not", "cannot")
    text = special_chars.sub(' ', text)
    text = re.sub(' +', ' ', text)
    return text


def load_data(filename, clean=True, encoding='utf-8'):
    """Read data from file into list of tuples"""
    dataset = []
    labels = set()
    with codecs.open(filename, 'r', encoding=encoding) as f:
        for line in f:
            if encoding is not 'utf-8':
                line = line.encode('utf-8').decode(encoding)  # convert string to utf-8 version
            line = line.strip().split(' ')  # all the tokens and labels are split by __BLANKSPACE__
            if clean:
                sentence = clean_text(' '.join(line[1:])).split(' ')  # clean text and convert to tokens again
            else:
                sentence = line[1:]
            label = int(line[0])
            labels.add(label)
            dataset.append((sentence, label))
    return dataset, len(labels)


def load_glove_vocab(filename):
    """Read word vocabulary from embeddings"""
    with open(filename, 'r', encoding='utf-8') as f:
        vocab = {line.strip().split()[0] for line in tqdm(f, desc='Loading GloVe vocabularies')}
    print('\t -- totally {} tokens in GloVe embeddings.\n'.format(len(vocab)))
    return vocab


def save_filtered_vectors(vocab, glove_path, save_path, word_dim):
    """Prepare pre-trained word embeddings for dataset"""
    embeddings = np.zeros([len(vocab), word_dim])  # embeddings[0] for PAD
    scale = np.sqrt(3.0 / word_dim)
    embeddings[1] = np.random.uniform(-scale, scale, [1, word_dim])  # for UNK
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Filtering GloVe embeddings'):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                idx = vocab[word]
                embeddings[idx] = np.asarray(embedding)
    sys.stdout.write('Saving filtered embeddings...')
    np.savez_compressed(save_path, embeddings=embeddings)
    sys.stdout.write(' done.\n')


def write_vocab(vocab, filename):
    """write vocabulary to file"""
    sys.stdout.write('Writing vocab to {}...'.format(filename))
    with open(filename, 'w') as f:
        for i, word in enumerate(vocab):
            f.write('{}\n'.format(word)) if i < len(vocab) - 1 else f.write(word)
    sys.stdout.write(' done. Totally {} tokens.\n'.format(len(vocab)))


def load_vocab(filename):
    """read vocabulary from file into dict"""
    word_idx = dict()
    idx_word = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            word_idx[word] = idx
            idx_word[idx] = word
    return word_idx, idx_word


def build_vocab(datasets, threshold=0):
    """Build word and char vocabularies"""
    word_count = dict()
    char_vocab = set()
    for dataset in datasets:
        for words, _ in dataset:
            for word in words:
                char_vocab.update(word)  # update char vocabulary
                word_count[word] = word_count.get(word, 0) + 1  # update word count in word dict
    word_count = reversed(sorted(word_count.items(), key=operator.itemgetter(1)))
    word_vocab = set([w[0] for w in word_count if w[1] >= threshold])
    char_vocab = char_vocab
    return word_vocab, char_vocab


def dump_to_json(dataset, filename):
    """Save built dataset into json"""
    if dataset is not None:
        with open(filename, 'w') as f:
            json.dump(dataset, f)
        sys.stdout.write('dump dataset to {}.\n'.format(filename))


def split_train_dev_test(dataset, dev_ratio=0.1, test_ratio=0.1, build_test=True, shuffle=True):
    """Split dataset into train, dev as well as test sets"""
    if shuffle:
        np.random.shuffle(dataset)
    data_size = len(dataset)
    if build_test:
        train_position = int(data_size * (1 - dev_ratio - test_ratio))
        dev_position = int(data_size * (1 - test_ratio))
        train_set = dataset[:train_position]
        dev_set = dataset[train_position:dev_position]
        test_set = dataset[dev_position:]
        return train_set, dev_set, test_set
    else:
        # dev_ratio = dev_ratio + test_ratio
        train_position = int(data_size * (1 - dev_ratio))
        train_set = dataset[:train_position]
        dev_set = dataset[train_position:]
        return train_set, dev_set, None


def fit_word_to_id(word, word_vocab, char_vocab):
    """Convert word str to word index and char indices"""
    char_ids = []
    for char in word:
        char_ids += [char_vocab[char]] if char in char_vocab else [char_vocab[UNK]]
    word = word_vocab[word] if word in word_vocab else word_vocab[UNK]
    return word, char_ids


def build_dataset(raw_dataset, filename, word_vocab, char_vocab, num_labels, one_hot=True):
    """Convert dataset into word/char index, make labels to be one hot vectors and dump to json file"""
    dataset = []
    for sentence, label in raw_dataset:
        words = []
        for word in sentence:
            words += [fit_word_to_id(word, word_vocab, char_vocab)]
        if one_hot:
            label = [1 if i == label else 0 for i in range(num_labels)]
        dataset.append({'sentence': words, 'label': label})
    dump_to_json(dataset, filename=filename)


def prepro_general(train_set, dev_set, test_set, num_labels, data_folder, glove_vocab, glove_path):
    """Performs to build vocabularies and processed dataset"""
    # build vocabularies
    word_vocab, char_vocab = build_vocab([train_set, dev_set])  # only process train and dev sets
    if glove_vocab is None:
        glove_vocab = load_glove_vocab(glove_path)
    word_vocab = [PAD, UNK] + list(word_vocab & glove_vocab)  # distinct vocab and add PAD and UNK tokens
    write_vocab(word_vocab, filename=os.path.join(data_folder, 'words.vocab'))
    char_vocab = [PAD, UNK] + list(char_vocab)  # add PAD and UNK tokens
    write_vocab(char_vocab, filename=os.path.join(data_folder, 'chars.vocab'))
    # build embeddings
    word_vocab, _ = load_vocab(os.path.join(data_folder, 'words.vocab'))
    save_filtered_vectors(word_vocab, glove_path, os.path.join(data_folder, 'glove.filtered.npz'), word_dim=300)
    # build dataset
    char_vocab, _ = load_vocab(os.path.join(data_folder, 'chars.vocab'))
    build_dataset(train_set, os.path.join(data_folder, 'train.json'), word_vocab, char_vocab, num_labels=num_labels,
                  one_hot=True)
    build_dataset(dev_set, os.path.join(data_folder, 'dev.json'), word_vocab, char_vocab, num_labels=num_labels,
                  one_hot=True)
    build_dataset(test_set, os.path.join(data_folder, 'test.json'), word_vocab, char_vocab, num_labels=num_labels,
                  one_hot=True)
    # save number of labels information into json
    with open(os.path.join(data_folder, 'label.json'), 'w') as f:
        json.dump({"label_size": num_labels}, f)


def prepro_sst(glove_path, glove_vocab=None, mode=1):
    print('Process sst{} dataset...'.format(mode))
    data_folder = os.path.join(target_dir, 'sst{}'.format(mode))
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # load dataset
    name = 'fine' if mode == 1 else 'binary'
    train_set, num_labels = load_data(os.path.join(source_dir, 'sst{}'.format(mode), 'stsa.{}.train'.format(name)))
    dev_set, _ = load_data(os.path.join(source_dir, 'sst{}'.format(mode), 'stsa.{}.dev'.format(name)))
    test_set, _ = load_data(os.path.join(source_dir, 'sst{}'.format(mode), 'stsa.{}.test'.format(name)))
    # build general
    prepro_general(train_set, dev_set, test_set, num_labels, data_folder, glove_vocab, glove_path)
    print()


def prepro_trec(glove_path, glove_vocab=None):
    print('Process trec dataset...')
    data_folder = os.path.join(target_dir, 'trec')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # load dataset
    train_set, num_labels = load_data(os.path.join(source_dir, 'trec', 'TREC.train.all'), encoding='windows-1252')
    test_set, _ = load_data(os.path.join(source_dir, 'trec', 'TREC.test.all'))
    train_set, dev_set, _ = split_train_dev_test(train_set, build_test=False)
    # build general
    prepro_general(train_set, dev_set, test_set, num_labels, data_folder, glove_vocab, glove_path)
    print()


def prepro_other(name, filename, glove_path, glove_vocab=None, encoding='utf-8'):
    print('Process {} dataset...'.format(name))
    data_folder = os.path.join(target_dir, name)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    # load dataset
    train_set, num_labels = load_data(os.path.join(source_dir, name, filename), encoding=encoding)
    train_set, dev_set, test_set = split_train_dev_test(train_set)
    # build general
    prepro_general(train_set, dev_set, test_set, num_labels, data_folder, glove_vocab, glove_path)
    print()


def main():
    glove_path = os.path.join(emb_dir, 'glove.42B.300d.txt')
    glove_vocab = load_glove_vocab(glove_path)
    # process sst1 dataset
    prepro_sst(glove_path, glove_vocab, mode=1)
    # process sst2 dataset
    prepro_sst(glove_path, glove_vocab, mode=2)
    # process trec dataset
    prepro_trec(glove_path, glove_vocab)
    # process subj dataset
    prepro_other('subj', 'subj.all', glove_path, glove_vocab, encoding='windows-1252')
    # process mr dataset
    prepro_other('mr', 'rt-polarity.all', glove_path, glove_vocab, encoding='windows-1252')
    # process mpqa dataset
    prepro_other('mpqa', 'mpqa.all', glove_path, glove_vocab)
    # process cr dataset
    prepro_other('cr', 'custrev.all', glove_path, glove_vocab)
    print('Pre-processing all the datasets finished... data is located at {}'.format(target_dir))


if __name__ == '__main__':
    main()
