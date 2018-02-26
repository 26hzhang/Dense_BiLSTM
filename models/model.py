import tensorflow as tf
import os
import numpy as np
from utils import get_logger, Progbar, batch_iter, pad_sequences
from models import multi_conv1d, highway_network, dropout, BiRNN, dense

np.random.seed(12345)


class DenseConnectBiLSTM(object):
    def __init__(self, config, resume_training=True, model_name='dense_bi_lstm'):
        # set configurations
        self.cfg, self.model_name, self.resume_training, self.start_epoch = config, model_name, resume_training, 1
        self.logger = get_logger(os.path.join(self.cfg.ckpt_path, 'log.txt'))
        # build model
        self._add_placeholder()
        self._add_embedding_lookup()
        self._build_model()
        self._add_loss_op()
        self._add_accuracy_op()
        self._add_train_op()
        print('params number: {}'.format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        # initialize model
        self.sess, self.saver = None, None
        self.initialize_session()

    def initialize_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=self.cfg.max_to_keep)
        self.sess.run(tf.global_variables_initializer())
        if self.resume_training:
            checkpoint = tf.train.get_checkpoint_state(self.cfg.ckpt_path)
            if not checkpoint:
                r = input("No checkpoint found in directory %s, cannot resume training. Do you want to start a new "
                          "training session?\n(y)es | (n)o: " % self.cfg.ckpt_path)
                if r.startswith('y'):
                    return
                else:
                    exit(0)
            print('Resume training from %s...' % self.cfg.ckpt_path)
            ckpt_path = checkpoint.model_checkpoint_path
            self.start_epoch = int(ckpt_path.split('-')[-1]) + 1
            print('Start Epoch: ', self.start_epoch)
            self.saver.restore(self.sess, ckpt_path)

    def restore_last_session(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.cfg.ckpt_path)  # get checkpoint state
        if ckpt and ckpt.model_checkpoint_path:  # restore session
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save_session(self, epoch):
        if not os.path.exists(self.cfg.ckpt_path):
            os.makedirs(self.cfg.dir_model)
        self.saver.save(self.sess, self.cfg.ckpt_path + self.model_name, global_step=epoch)

    def close_session(self):
        self.sess.close()

    def _add_placeholder(self):
        # shape = (batch_size, max_sentence_length)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        # shape = (batch_size)
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
        # shape = (batch_size, max_sentence_length, max_word_length)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name='char_ids')
        # shape = (batch_size, max_sentence_length)
        self.word_len = tf.placeholder(tf.int32, shape=[None, None], name='word_len')
        # shape = (batch_size, label_size)
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        # hyper-parameters
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

    def _get_feed_dict(self, words, labels=None, lr=None, is_train=None):
        word_ids, char_ids = zip(*words)
        word_ids, seq_len = pad_sequences(word_ids, max_length=None, pad_tok=0, nlevels=1)
        feed_dict = {self.word_ids: word_ids, self.seq_len: seq_len}
        if self.cfg.use_char_emb:
            char_ids, word_len = pad_sequences(char_ids, max_length=None, pad_tok=0, max_length_2=None, nlevels=2)
            feed_dict[self.char_ids] = char_ids
            feed_dict[self.word_len] = word_len
        if labels is not None:
            feed_dict[self.labels] = labels
        if lr is not None:
            feed_dict[self.lr] = lr
        if is_train is not None:
            feed_dict[self.is_train] = is_train
        return feed_dict

    def _add_embedding_lookup(self):
        with tf.variable_scope('word_embeddings'):
            if self.cfg.use_word_emb:
                _word_emb = tf.Variable(self.cfg.word_emb, name='_word_emb', trainable=self.cfg.finetune_emb,
                                        dtype=tf.float32)
            else:
                _word_emb = tf.get_variable(name='_word_emb', shape=[self.cfg.vocab_size, self.cfg.word_dim],
                                            trainable=True, dtype=tf.float32)
            word_emb = tf.nn.embedding_lookup(_word_emb, self.word_ids, name='word_emb')

        if self.cfg.use_char_emb:  # use cnn to generate chars representation
            with tf.variable_scope('char_embeddings'):
                _char_emb = tf.get_variable(name='_char_emb', dtype=tf.float32, trainable=True,
                                            shape=[self.cfg.char_vocab_size, self.cfg.char_dim])
                char_emb = tf.nn.embedding_lookup(_char_emb, self.char_ids, name='char_emb')
                char_emb_shape = tf.shape(char_emb)
                char_rep = multi_conv1d(char_emb, self.cfg.filter_sizes, self.cfg.heights, "VALID",  self.is_train,
                                        self.cfg.keep_prob, scope="char_cnn")
                char_rep = tf.reshape(char_rep, [char_emb_shape[0], char_emb_shape[1], self.cfg.char_rep_dim])
                word_emb = tf.concat([word_emb, char_rep], axis=-1)  # concat word emb and corresponding char rep
        if self.cfg.use_highway:
            self.word_emb = highway_network(word_emb, self.cfg.highway_num_layers, bias=True, is_train=self.is_train,
                                            keep_prob=self.cfg.keep_prob)
        else:
            self.word_emb = dropout(word_emb, keep_prob=self.cfg.keep_prob, is_train=self.is_train)
        print('word embedding shape: {}'.format(self.word_emb.get_shape().as_list()))

    def _build_model(self):
        with tf.variable_scope('dense_connect_bi_lstm'):
            # create dense connected bi-lstm layers
            dense_bi_lstm = []
            for idx in range(self.cfg.num_layers):
                if idx < self.cfg.num_layers - 1:
                    dense_bi_lstm.append(BiRNN(num_units=self.cfg.num_units, scope='bi_lstm_layer_{}'.format(idx)))
                else:
                    dense_bi_lstm.append(BiRNN(num_units=self.cfg.num_units_last, scope='bi_lstm_layer_{}'.format(idx)))
            # processing data
            cur_inputs = self.word_emb
            for idx in range(self.cfg.num_layers):
                cur_rnn_outputs = dense_bi_lstm[idx](cur_inputs, seq_len=self.seq_len)
                if idx < self.cfg.num_layers - 1:
                    cur_inputs = tf.concat([cur_inputs, cur_rnn_outputs], axis=-1)
                else:
                    cur_inputs = cur_rnn_outputs
            dense_bi_lstm_outputs = cur_inputs
            print('dense bi-lstm outputs shape: {}'.format(dense_bi_lstm_outputs.get_shape().as_list()))

        with tf.variable_scope('average_pooling'):
            # according to the paper (https://arxiv.org/pdf/1802.00889.pdf) description in P4, simply compute average ?
            avg_pool_outputs = tf.reduce_mean(dense_bi_lstm_outputs, axis=1)
            avg_pool_outputs = dropout(avg_pool_outputs, keep_prob=self.cfg.keep_prob, is_train=self.is_train)
            print('average pooling outputs shape: {}'.format(avg_pool_outputs.get_shape().as_list()))

        with tf.variable_scope('project', regularizer=tf.contrib.layers.l2_regularizer(self.cfg.l2_reg)):
            self.logits = dense(avg_pool_outputs, self.cfg.label_size, use_bias=True, scope='dense')
            print('logits shape: {}'.format(self.logits.get_shape().as_list()))

    def _add_loss_op(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        '''if self.cfg.l2_reg is not None and self.cfg.l2_reg > 0.0:
            # l2 constraints over softmax parameters (i.e., project parameters) ?
            train_vars = tf.trainable_variables(scope='project')
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars if 'bias' not in v.name])
            self.loss = tf.reduce_mean(loss + self.cfg.l2_reg * l2_loss)
        else:
            self.loss = tf.reduce_mean(loss)'''
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(loss) + 0.5 * l2_loss

    def _add_accuracy_op(self):
        self.predicts = tf.argmax(self.logits, axis=-1)
        self.actuals = tf.argmax(self.labels, axis=-1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicts, self.actuals), dtype=tf.float32))

    def _add_train_op(self):
        with tf.variable_scope('train_step'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            if self.cfg.grad_clip is not None:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)

    def train(self, trainset, devset, testset, batch_size=64, epochs=50, shuffle=True):
        self.logger.info('Start training...')
        init_lr = self.cfg.lr  # initial learning rate, used for decay learning rate
        best_score = 0.0  # record the best score
        best_score_epoch = 1  # record the epoch of the best score obtained
        no_imprv_epoch = 0  # no improvement patience counter
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info('Epoch %2d/%2d:' % (epoch, epochs))
            progbar = Progbar(target=(len(trainset) + batch_size - 1) // batch_size)  # number of batches
            if shuffle:
                np.random.shuffle(trainset)  # shuffle training dataset each epoch
            # training each epoch
            for i, (words, labels) in enumerate(batch_iter(trainset, batch_size)):
                feed_dict = self._get_feed_dict(words, labels, lr=self.cfg.lr, is_train=True)
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                progbar.update(i + 1, [("train loss", train_loss)])
            if devset is not None:
                self.evaluate(devset, batch_size)
            cur_score = self.evaluate(testset, batch_size, is_devset=False)
            # learning rate decay
            if self.cfg.decay_lr:
                self.cfg.lr = init_lr / (1 + self.cfg.lr_decay * epoch)
            # performs model saving and evaluating on test dataset
            if cur_score > best_score:
                no_imprv_epoch = 0
                self.save_session(epoch)
                best_score = cur_score
                best_score_epoch = epoch
                self.logger.info(' -- new BEST score on TEST dataset: {:05.3f}'.format(best_score))
            else:
                no_imprv_epoch += 1
                if no_imprv_epoch >= self.cfg.no_imprv_patience:
                    self.logger.info('early stop at {}th epoch without improvement for {} epochs, BEST score: '
                                     '{:05.3f} at epoch {}'.format(epoch, no_imprv_epoch, best_score, best_score_epoch))
                    break
        self.logger.info('Training process done...')

    def evaluate(self, dataset, batch_size, is_devset=True):
        accuracies = []
        for words, labels in batch_iter(dataset, batch_size):
            feed_dict = self._get_feed_dict(words, labels, lr=None, is_train=False)
            accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
            accuracies.append(accuracy)
        acc = np.mean(accuracies) * 100
        self.logger.info("Testing model over {} dataset: accuracy - {:05.3f}".format('DEVELOPMENT' if is_devset else
                                                                                     'TEST', acc))
        return acc
