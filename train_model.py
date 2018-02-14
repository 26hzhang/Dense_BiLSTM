from utils import load_json
from models import Config, DenseConnectBiLSTM
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings


def main():
    data_folder = os.path.join('.', 'dataset', 'data')
    # set tasks
    task = 'sst2'
    source_dir = os.path.join(data_folder, task)
    # create config
    config = Config(source_dir)
    # load datasets
    trainset = load_json(os.path.join(source_dir, 'train.json'))
    devset = load_json(os.path.join(source_dir, 'dev.json'))
    testset = load_json(os.path.join(source_dir, 'test.json'))
    # build model
    model = DenseConnectBiLSTM(config)
    # training
    batch_size = 32
    epochs = 30
    model.train(trainset, devset, testset, batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    main()
