from utils import load_json
from models import Config, DenseConnectBiLSTM
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings


def main(task, resume):
    data_folder = os.path.join('.', 'dataset', 'data')
    # set tasks
    source_dir = os.path.join(data_folder, task)
    # create config
    config = Config(task)
    # load datasets
    trainset = load_json(os.path.join(source_dir, 'train.json'))
    devset = load_json(os.path.join(source_dir, 'dev.json'))
    testset = load_json(os.path.join(source_dir, 'test.json'))
    # build model
    model = DenseConnectBiLSTM(config, resume_training=resume)
    # training
    batch_size = 200
    epochs = 30
    model.train(trainset, devset, testset, batch_size=batch_size, epochs=epochs)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_task = sys.argv[1]
    else:
        train_task = 'subj'  # default
    if len(sys.argv) > 2:
        resume_training = sys.argv[2]
    else:
        resume_training = True  # default
    main(train_task, resume_training)
