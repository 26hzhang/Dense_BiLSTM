from utils import load_json
from models import Config, DenseConnectBiLSTM
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings


def main():
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
    model = DenseConnectBiLSTM(config, resume_training=resume_training)
    # training
    batch_size = 200
    epochs = 30
    if has_devset:
        model.train(trainset, devset, testset, batch_size=batch_size, epochs=epochs, shuffle=True)
    else:
        trainset = trainset + devset
        model.train(trainset, None, testset, batch_size=batch_size, epochs=epochs, shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='set train task (cr|mpqa|mr|sst1|sst2|subj|trec).')
    parser.add_argument('--resume_training', type=str, required=True, help='resume previous trained model.')
    parser.add_argument('--has_devset', type=str, required=True, help='indicates if the task has development dataset.')
    args, _ = parser.parse_known_args()
    task = args.task
    resume_training = True if args.resume_training == 'True' else False
    has_devset = True if args.has_devset == 'True' else False
    main()
