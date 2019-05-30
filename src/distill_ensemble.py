import _pickle as cPickle
import argparse
from keras.layers import Dense, Input, Dropout, TimeDistributed
from models import CNNv2
from distill_datasets import load_all, Timer
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from src.autoencoder import *
import threading

__author__ = 'Bonggun Shin'


class TeacherLogitBase(object):
    def __init__(self, x, logits, gold, batch_size=1, n_teacher = 20, n_iter=10):
        self.batch_size = batch_size
        self.x = x
        self.logits = logits
        self.gold = gold
        self.steps = len(self.x) // batch_size
        self.lock = threading.Lock()
        # self.chunk_size = min(20000, self.steps * self.batch_size)  # ~len(self.x)
        self.chunk_size = len(self.x)
        self.n_teacher = n_teacher
        self.n_iter = n_iter
        self.init()

    def _softmax(self, x):
        e_x = np.exp(x)
        ret = e_x / e_x.sum(axis=0)
        return ret

    def _inverse_softmax(self, x):
        e_x = 1 / np.exp(x)
        ret = e_x / e_x.sum(axis=0)
        return ret

    def init(self):
        raise NotImplementedError()

    def _squash(self, x):
        s_squared_norm = np.sum(np.square(x))
        scale = s_squared_norm / (1 + s_squared_norm) / np.sqrt(s_squared_norm + np.finfo(float).eps)
        return scale * x

    def _generator(self):
        while True:
            indices = np.random.permutation(self.chunk_size)
            # x_perm = self.x[indices]
            # y_perm = self.weighted_logits[indices]
            # for i in range(0, self.chunk_size):
            #     irange = indices[i]
            for i in range(0, self.chunk_size, self.batch_size):
                irange = indices[i:i + self.batch_size]
                X = self.x[irange]
                y = self.weighted_logits[irange]
                # X = x_perm[i:i + self.batch_size]
                # y = y_perm[i:i + self.batch_size]

                yield (X, y)

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return self.generator.next()

    def __next__(self):
        return self.generator.__next__()


class TeacherLogitAverage(TeacherLogitBase):
    def init(self):
        self._init_avg()

    def _init_avg(self):
        self.generator = self._generator()

        self.weighted_logits = np.zeros(self.logits[0].shape)
        for i, y in enumerate(self.gold):
            target_logits = self.logits[:, i, :]
            self.weighted_logits[i, :] = np.average(target_logits, axis=0)


class TeacherLogitDSM(TeacherLogitBase):
    def init(self):
        self._init_dsm()

    def _init_dsm(self):  # dsm: doubly softmax
        self.generator = self._generator()

        attentions = np.zeros((len(self.gold), self.n_teacher))

        self.weighted_logits = np.zeros(self.logits[0].shape)
        for i, y in enumerate(self.gold):
            target_logits = self.logits[:, i, :]
            softmax_target_logits = np.zeros(target_logits.shape)
            for j, logit in enumerate(target_logits):
                softmax_target_logits[j, :] = self._softmax(logit)

            attentions[i, :] = self._softmax(softmax_target_logits[:, y])
            self.weighted_logits[i, :] = np.average(target_logits, axis=0, weights=attentions[i, :])


class TeacherLogitIDSM(TeacherLogitBase):
    def init(self):
        self._init_idsm()

    def _init_idsm(self):  # dsm: inverse softmax
        self.generator = self._generator()

        attentions = np.zeros((len(self.gold), self.n_teacher))

        self.weighted_logits = np.zeros(self.logits[0].shape)
        for i, y in enumerate(self.gold):
            target_logits = self.logits[:, i, :]
            softmax_target_logits = np.zeros(target_logits.shape)
            for j, logit in enumerate(target_logits):
                softmax_target_logits[j, :] = self._softmax(logit)

            attentions[i, :] = self._inverse_softmax(softmax_target_logits[:, y])
            self.weighted_logits[i, :] = np.average(target_logits, axis=0, weights=attentions[i, :])



class TeacherLogitRTPlus(TeacherLogitBase):
    def init(self):
        self._init_routing()

    def _init_routing(self):
        self.generator = self._generator()

        self.weighted_logits = np.zeros(self.logits[0].shape)

        for i, y in enumerate(self.gold):
            b = np.zeros(self.n_teacher)
            target_logits = self.logits[:, i, :]

            sqs_logits = np.zeros(target_logits.shape)
            for j, logit in enumerate(target_logits):
                sqs_logits[j] = self._squash(logit)

            for iter in range(self.n_iter):
                c = self._softmax(b)
                s = np.average(target_logits, axis=0, weights=c)
                v = self._squash(s)

                if iter < self.n_iter-1:
                    for j, logit in enumerate(target_logits):
                        b[j] = b[j] + np.dot(sqs_logits[j], v)

            self.weighted_logits[i, :] = s


class TeacherLogitRT(TeacherLogitBase):
    def init(self):
        self._init_routing()

    def _init_routing(self):
        self.generator = self._generator()

        self.weighted_logits = np.zeros(self.logits[0].shape)

        for i, y in enumerate(self.gold):
            b = np.zeros(self.n_teacher)
            target_logits = self.logits[:, i, :]

            sqs_logits = np.zeros(target_logits.shape)
            for j, logit in enumerate(target_logits):
                sqs_logits[j] = self._squash(logit)

            for iter in range(self.n_iter):
                c = self._softmax(b)
                s = np.average(target_logits, axis=0, weights=c)
                v = self._squash(s)

                if iter < self.n_iter-1:
                    for j, logit in enumerate(target_logits):
                        b[j] = b[j] - np.dot(sqs_logits[j], v)

            self.weighted_logits[i, :] = s


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.swapaxes(x, 0, 1)
    e_x = np.exp(x - np.max(x))
    ret = e_x / e_x.sum(axis=0)
    ret = np.swapaxes(ret, 0, 1)
    return ret


def evaluation_summary(student_model, x_trn, y_trn, x_dev, y_dev, x_tst, y_tst, eval_name):
    score_list = []
    y_trn_student_softmax_prediction = softmax(np.array(student_model.predict(x_trn, batch_size=256, verbose=0)))
    y_trn_student_prediction = np.argmax(y_trn_student_softmax_prediction, axis=1)
    score = sum(y_trn_student_prediction == y_trn) * 1.0 / len(y_trn)
    score_list.append(score)

    y_dev_student_softmax_prediction = softmax(np.array(student_model.predict(x_dev, batch_size=256, verbose=0)))
    y_dev_student_prediction = np.argmax(y_dev_student_softmax_prediction, axis=1)
    score = sum(y_dev_student_prediction == y_dev) * 1.0 / len(y_dev)
    score_list.append(score)

    y_tst_student_softmax_prediction = softmax(np.array(student_model.predict(x_tst, batch_size=256, verbose=0)))
    y_tst_student_prediction = np.argmax(y_tst_student_softmax_prediction, axis=1)
    score = sum(y_tst_student_prediction == y_tst) * 1.0 / len(y_tst)
    score_list.append(score)

    return score_list


class MyCallback(ModelCheckpoint):
    def __init__(self, filepath, data, real_save=True, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        print('my callback init')
        super(MyCallback, self).__init__(filepath, monitor, verbose,
                                         save_best_only, save_weights_only,
                                         mode, period)

        self.x_trn, self.y_trn, self.x_dev, self.y_dev, self.x_tst, self.y_tst = data
        self.score_trn = 0
        self.score_dev = 0
        self.score_tst = 0
        self.real_save = real_save

    def evaluate(self):
        score_list = evaluation_summary(self.model, self.x_trn, self.y_trn, self.x_dev, self.y_dev,
                                        self.x_tst, self.y_tst, '[Epoch]')

        if self.score_dev < score_list[1]:
            self.score_trn = score_list[0]
            self.score_dev = score_list[1]
            self.score_tst = score_list[2]

            print("\nupdated!!")
            if self.real_save == True:
                if self.save_weights_only:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)

        print('\n[This Epoch]')
        print('\t'.join(map(str, [score_list[0], score_list[1], score_list[2]])))
        print('[Current Best]')
        print('\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst])))

    def on_train_end(self, logs=None):
        print('[Best:on_train_end]')
        print('\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst])))

    def on_epoch_end(self, epoch, logs=None):
        self.evaluate()


def custom_loss_Ba(y_true, y_pred):
    # Ba, J. and Caruana, R. Do deep nets really need to be deep? In NIPS 2014.
    # tf_loss = tf.nn.l2_loss(teacher - student)/batch_size
    t = y_true - y_pred
    rval = K.mean(K.square(t))
    return rval
    # objective = K.categorical_crossentropy(output=student, target=y_gold2)
    # return objective


def StudentCommon(model_input, max_len, dataset_name, pretrained_projection_layer=None):
    teacher_w2vdim = 400
    student_w2vdim = 50
    student_filter_sizes = (2, 3, 4, 5)
    student_num_filters = 32

    student_hidden_dims = 50

    dropout_prob = 0.1

    projection_layer1 = Dense(teacher_w2vdim, weights=pretrained_projection_layer[0:2], activation="relu")
    projection_layer2 = Dense(student_w2vdim, weights=pretrained_projection_layer[2:], activation=None)

    z = Dropout(dropout_prob)(model_input)
    z = TimeDistributed(projection_layer1, input_shape=(max_len, teacher_w2vdim))(z)
    Wz = TimeDistributed(projection_layer2)(z)


    model = CNNv2(model_input, Wz, student_filter_sizes, student_num_filters, dropout_prob, student_hidden_dims,
                  dataset_name, activation=None)
    return model


def get_logits(logit_dic):
    logit_trn = []
    logit_dev = []
    logit_tst = []

    for model_name in ['cnn2', 'lstm']:
        for teacher_attempt in range(10 ):
            data = logit_dic['%s_%d' % (model_name, teacher_attempt)]
            logit_trn.append(data[0])
            logit_dev.append(data[1])
            logit_tst.append(data[2])

    return np.array(logit_trn), np.array(logit_dev), np.array(logit_tst)


def run(method, gpunum, dataset_name, attempt):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum

    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())

    with Timer("loading..."):
        logit_dic = cPickle.load(open('../data/logit_dic_%s.pkl' % (dataset_name), 'rb'))

        logit_trn, logit_dev, logit_tst = get_logits(logit_dic)


    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features, maxlen = \
            load_all(400, dataset_name=dataset_name)
        x_trn = embedding[x_trn]
        x_dev = embedding[x_dev]
        x_tst = embedding[x_tst]

    print('d')
    ae_name_list = {"mr": "news", "sstb": "sst5", "sst5": "sst5", "subj": "news", "trec": "news",
                    "cr": "sst5", "mpqa": "news", "s17": "s17"}

    pretrained=True
    projection_model = 'deep'
    teacher_w2vdim = 400
    batch_size = 200
    epochs = 200

    model_directory = '../model_distill_ba'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    filepath_distill_ensemble_model = '%s/%s-ensemble-%d' % (model_directory, dataset_name, attempt)

    print('distill_ensemble_model=%s' % filepath_distill_ensemble_model)
    autoenc = DeepAutoencoder([400, 400, 50])
    filepath_ae = '../model_ae/selu_deep-%s-%d-%d' % (ae_name_list[dataset_name], 400, 50)
    print('filepath_ae=%s' % filepath_ae)
    autoenc.autoencoder.load_weights(filepath_ae)
    pretrained_projection_layer = autoenc.autoencoder.get_weights()
    pretrained_projection_layer = pretrained_projection_layer[0:4]

    print('loading ae model (%s)' % filepath_ae)

    input_shape = (maxlen, teacher_w2vdim)
    model_input = Input(shape=input_shape)
    student_model = StudentCommon(model_input, maxlen, dataset_name, pretrained_projection_layer=pretrained_projection_layer)
    student_model.compile(loss=custom_loss_Ba, optimizer="adam")
    data = tuple((x_trn, y_trn, x_dev, y_dev, x_tst, y_tst))
    checkpoint = MyCallback(filepath_distill_ensemble_model, data, real_save=True, monitor='val_loss', verbose=1,
                            save_best_only=True,
                            mode='auto')

    callbacks_list = [checkpoint]

    # if loss_name == 'bharat':
    # student_model.fit(x_trn, logit_trn,
    #                   batch_size=batch_size,
    #                   callbacks=callbacks_list,
    #                   epochs=epochs,
    #                   validation_data=(x_dev, logit_dev))

    if method=='avg':
        data_gen_trn = TeacherLogitAverage(x_trn, logit_trn, y_trn, batch_size=batch_size)
        data_gen_dev = TeacherLogitAverage(x_dev, logit_dev, y_dev, batch_size=batch_size)

    elif method=='dsm':
        data_gen_trn = TeacherLogitDSM(x_trn, logit_trn, y_trn, batch_size=batch_size)
        data_gen_dev = TeacherLogitDSM(x_dev, logit_dev, y_dev, batch_size=batch_size)

    elif method=='idsm':
        data_gen_trn = TeacherLogitIDSM(x_trn, logit_trn, y_trn, batch_size=batch_size)
        data_gen_dev = TeacherLogitIDSM(x_dev, logit_dev, y_dev, batch_size=batch_size)

    elif method == 'rtplus':
        data_gen_trn = TeacherLogitRTPlus(x_trn, logit_trn, y_trn, batch_size=batch_size, n_iter=10)
        data_gen_dev = TeacherLogitRTPlus(x_dev, logit_dev, y_dev, batch_size=batch_size, n_iter=10)

    elif method=='rt10':
        data_gen_trn = TeacherLogitRT(x_trn, logit_trn, y_trn, batch_size=batch_size, n_iter=10)
        data_gen_dev = TeacherLogitRT(x_dev, logit_dev, y_dev, batch_size=batch_size, n_iter=10)

    elif method == 'rt100':
        data_gen_trn = TeacherLogitRT(x_trn, logit_trn, y_trn, batch_size=batch_size, n_iter=100)
        data_gen_dev = TeacherLogitRT(x_dev, logit_dev, y_dev, batch_size=batch_size, n_iter=100)

    else:
        data_gen_trn = TeacherLogitDSM(x_trn, logit_trn, y_trn, batch_size=batch_size)
        data_gen_dev = TeacherLogitDSM(x_dev, logit_dev, y_dev, batch_size=batch_size)


    student_model.fit_generator(generator=data_gen_trn,
                                steps_per_epoch=data_gen_trn.steps,
                                validation_data=data_gen_dev,
                                validation_steps=data_gen_dev.steps,
                                epochs=epochs,
                                callbacks=callbacks_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default=0, choices=range(20), type=int)
    parser.add_argument('-m', default="rtplus", choices=["avg", "dsm", "idsm", "rt10", "rt100", "rtplus"], type=str) # dsm: double softmax
    parser.add_argument('-g', default="0", choices=["0", "1", "2", "3"], type=str)
    parser.add_argument('-ds', default="sst5", choices=["mr", "sstb", "sst5", "subj", "trec", "cr", "mpqa", "s17"],
                        type=str)

    args = parser.parse_args()

    run(args.m, args.g, args.ds, args.t)
