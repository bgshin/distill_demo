from keras.layers import Dense, Lambda, Input, Dropout, TimeDistributed, Activation
from keras.layers.merge import Multiply, Add
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import backend as K
import argparse
from src.autoencoder import *
from distill_datasets import load_all, Timer, maxlen_list
from models import CNNv1, CNNv2, LSTMv1

__author__ = 'Bonggun Shin'


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
        #
        # print '[Best]'
        # print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))

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


def custom_loss_Bharat(std, num_class):
    def loss_Bharat(y_true, y_pred):
        # Bharat Bhusan Sau Vineeth N. Balasubramanian,
        # Deep Model Compression: Distilling Knowledge from Noisy Teachers. arXiv 2016.

        const_tensor = K.ones(shape=[1, num_class])
        gaussian = K.random_normal(shape=[1, num_class], mean=0., stddev=std)

        noise = Add()([const_tensor, gaussian])
        noise = K.tile(noise, K.shape(K.expand_dims(y_pred, 1))[0:2])

        y_true = Multiply()([y_true, noise])

        t = y_true - y_pred
        rval = K.mean(K.square(t))

        return rval

    return loss_Bharat


def custom_loss_Hinton(y_gold, tau, alpha, num_class):
    def loss_hinton(y_true, y_pred):
        # y_true = y_true[0]
        # tau = 1.5
        # alpha =0.1
        # Hinton, G. E., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. arXiv 2015.
        teacher_tau = Activation('softmax')(y_true / tau)
        student_tau = Activation('softmax')(y_pred / tau)

        # teacher = Activation('softmax')(y_true)
        student = Activation('softmax')(y_pred)

        # y_gold2 = K.squeeze(K.one_hot(K.cast(y_gold, tf.int32), num_class), axis=1)
        y_gold2 = K.one_hot(K.cast(y_gold, tf.int32), num_class)

        objective1 = K.categorical_crossentropy(output=student, target=y_gold2)
        objective2 = K.categorical_crossentropy(output=student_tau, target=teacher_tau)

        return alpha*objective1 + (1-alpha)*objective2

    return loss_hinton

def custom_loss_Mou(y_gold, num_class):
    def loss_Mou(y_true, y_pred):
        # Lili Mou, CIKM, 2016
        student = Activation('softmax')(y_pred)
        # y_gold2 = K.squeeze(K.one_hot(K.cast(y_gold, tf.int32), num_class), axis=1)
        y_gold2 = K.one_hot(K.cast(y_gold, tf.int32), num_class)

        objective = K.categorical_crossentropy(output=student, target=y_gold2)
        return objective

    return loss_Mou



def run(attempt, gpunum, loss_name, real_save, tau, alpha, std, projection_model, pretrained, dataset_name, model_name,
        student_w2vdim):
    # set parameters:
    if model_name=='lstm':
        best_attempt_list = {"mr": 4, "sstb": 5, "sst5": 4, "subj": 5, "trec": 2, "cr": 5, "mpqa": 1, "s17": 3}
    else: # cnn
        best_attempt_list = {"mr": 5, "sstb": 7, "sst5": 0, "subj": 4, "trec": 8, "cr": 8, "mpqa": 3, "s17": 3}

    ae_name_list = {"mr": "news", "sstb": "sst5", "sst5": "sst5", "subj": "news", "trec": "news",
                    "cr": "sst5", "mpqa": "news", "s17": "s17"}

    if pretrained is True:
        print('======================== pretrained projection ========================')
    else:
        print('======================== random projection ========================')


    teacher_w2vdim = 400
    filter_sizes = (2, 3, 4, 5)
    student_filter_sizes = (2, 3, 4, 5)
    num_filters = 32
    student_num_filters = 32
    hidden_dims = 50
    student_hidden_dims = 50

    # maxlen = maxlen_list[dataset_name]
    batch_size = 200
    epochs = 200
    dropout_prob = 0.1

    if dataset_name == 'sst5':
        num_class = 5
    elif dataset_name == 'sstb' or dataset_name == 'mpqa' or dataset_name == 'mr' or dataset_name == 'cr' \
            or dataset_name == 'subj':
        num_class = 2
    elif dataset_name == 'trec':
        num_class = 6
    elif dataset_name == 's17':
        num_class = 3
    else:
        num_class = 2

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum
    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())



    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features, maxlen = \
            load_all(teacher_w2vdim, dataset_name=dataset_name)

        if model_name is not "cnn1":
            x_trn = embedding[x_trn]
            x_dev = embedding[x_dev]
            x_tst = embedding[x_tst]

        print("==================================== [%s]maxlen=%d ====================================" \
              % (dataset_name, maxlen))


    def StudentCommon(model_input, teacher_embedding, projection_model, max_len, pretrained_projection_layer=None):
        txt_emb = Lambda(lambda x: x[:, :, :teacher_embedding],
                         output_shape=(max_len, teacher_embedding))(model_input)

        y_gold = Lambda(lambda x: x[:, 0, teacher_embedding],
                        output_shape=(1, 1))(model_input)

        # y_gold = K.squeeze(K.squeeze(y_gold, axis=-1), axis=-1)

        if projection_model == 'vanila':
            if pretrained_projection_layer is None:
                projection_layer = Dense(student_w2vdim, activation=None)
            else:
                projection_layer = Dense(student_w2vdim, weights=pretrained_projection_layer[0:2], activation="relu")
            Wz = TimeDistributed(projection_layer)(txt_emb)

        elif projection_model == 'deep':
            if pretrained_projection_layer is None:
                projection_layer1 = Dense(teacher_w2vdim, activation="relu")
                projection_layer2 = Dense(student_w2vdim, activation=None)
            else:
                projection_layer1 = Dense(teacher_w2vdim, weights=pretrained_projection_layer[0:2], activation="relu")
                projection_layer2 = Dense(student_w2vdim, weights=pretrained_projection_layer[2:], activation=None)

            z = Dropout(dropout_prob)(txt_emb)
            z = TimeDistributed(projection_layer1, input_shape=(max_len, teacher_w2vdim))(z)
            Wz = TimeDistributed(projection_layer2)(z)


        else:
            print('wrong model name (%s)' % projection_model)
            exit()

        # (model_input, filter_sizes, num_filters, dropout_prob, hidden_dims, dataset_name,
        #  path_trained_weights = None, student = False):
        model = CNNv2(model_input, Wz, student_filter_sizes, student_num_filters, dropout_prob, student_hidden_dims,
                      dataset_name, activation=None)
        return model, y_gold


    with Timer("Build Teacher Model..."):
        input_shape = (maxlen,teacher_w2vdim)
        model_input = Input(shape=input_shape)

        best_attempt = best_attempt_list[dataset_name]
        teacher_model_path = '../model/teacher-%s-%s-%d-%d' % (dataset_name, model_name, teacher_w2vdim, best_attempt)
        print('teacher_model_path=%s' % teacher_model_path)

        if model_name == "cnn2":
            teacher_model = CNNv2(model_input, model_input, filter_sizes, num_filters, dropout_prob, hidden_dims, dataset_name,
                      path_trained_weights=teacher_model_path, activation=None)

        elif model_name =="lstm":
            teacher_model = LSTMv1(model_input, model_input, dropout_prob, hidden_dims, dataset_name,
                                   path_trained_weights=teacher_model_path, activation=None)

        else:
            print("wrong model_name(%s)" % model_name)
            exit()


        print(teacher_model.count_params())
        print(teacher_model.summary())


    with Timer("Get Teacher's Logits..."):
        y_trn_logit = np.array(teacher_model.predict(x_trn, verbose=1))
        y_dev_logit = np.array(teacher_model.predict(x_dev, verbose=1))
        y_tst_logit = np.array(teacher_model.predict(x_tst, verbose=1))

        x_trn_compound = np.concatenate((x_trn, np.expand_dims(np.repeat(np.expand_dims(y_trn, -1), maxlen, axis=1), -1)), axis=-1)
        x_dev_compound = np.concatenate((x_dev, np.expand_dims(np.repeat(np.expand_dims(y_dev, -1), maxlen, axis=1), -1)), axis=-1)
        x_tst_compound = np.concatenate((x_tst, np.expand_dims(np.repeat(np.expand_dims(y_tst, -1), maxlen, axis=1), -1)), axis=-1)

    K.clear_session()


    print(loss_name)
    print('projection_model=%s' % projection_model)
    if pretrained is True:
        if projection_model == 'vanila':
            autoenc = Autoencoder(400, student_w2vdim)
            filepath_ae = '../model_ae/selu_vanila-%s-%d-%d' % (
            ae_name_list[dataset_name], teacher_w2vdim, student_w2vdim)
            print('filepath_ae=%s' % filepath_ae)
            autoenc.autoencoder.load_weights(filepath_ae)
            pretrained_projection_layer = autoenc.autoencoder.get_weights()
            pretrained_projection_layer = pretrained_projection_layer[0:2]

        elif projection_model == 'deep':
            autoenc = DeepAutoencoder([400, 400, student_w2vdim])
            filepath_ae = '../model_ae/selu_deep-%s-%d-%d' % (ae_name_list[dataset_name], teacher_w2vdim, student_w2vdim)
            print('filepath_ae=%s' % filepath_ae)
            autoenc.autoencoder.load_weights(filepath_ae)
            pretrained_projection_layer = autoenc.autoencoder.get_weights()
            pretrained_projection_layer = pretrained_projection_layer[0:4]

        else:
            autoenc = DeepAutoencoder([400, 400, student_w2vdim])
            filepath_ae = '../model_ae/selu_deep-%s-%d-%d' % (ae_name_list[dataset_name], teacher_w2vdim, student_w2vdim)
            print('filepath_ae=%s' % filepath_ae)
            autoenc.autoencoder.load_weights(filepath_ae)
            pretrained_projection_layer = autoenc.autoencoder.get_weights()
            pretrained_projection_layer = pretrained_projection_layer[0:4]

        print('loading ae model (%s)' % filepath_ae)

    else:
        pretrained_projection_layer = None

    if loss_name == 'hinton':
        print('hin1')
        if real_save is True:
            model_directory = '../model_distill_hinton'
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

        filepath_distill_model = '%s/%s-%s-%s-%d-%d-%f-%d-%d-%d' % \
                                 (model_directory, dataset_name, model_name, projection_model, pretrained, tau, alpha, teacher_w2vdim,
                                  student_w2vdim, attempt)

        print('hinton distill_path=%s' % filepath_distill_model)

        input_shape = (maxlen, teacher_w2vdim+1)
        model_input = Input(shape=input_shape)

        student_model, y_gold = StudentCommon(model_input, teacher_w2vdim, projection_model, maxlen,
                                              pretrained_projection_layer=pretrained_projection_layer)
        student_model.compile(loss=custom_loss_Hinton(y_gold=y_gold, tau=tau, alpha=alpha, num_class=num_class),
                              optimizer="adam", metrics=["mse"])
        data = tuple((x_trn_compound, y_trn, x_dev_compound, y_dev, x_tst_compound, y_tst))
        checkpoint = MyCallback(filepath_distill_model, data, real_save=real_save, monitor='val_loss', verbose=1, save_best_only=True,
                                mode='auto')

    elif loss_name == 'ba':
        print('ba2')
        if real_save is True:
            model_directory = '../model_distill_ba'
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

        filepath_distill_model = '%s/%s-%s-%s-%d-%d-%d-%d' % \
                                 (model_directory,
                                 dataset_name, model_name, projection_model, pretrained, teacher_w2vdim, student_w2vdim,
                                 attempt)
        print('ba distill_path=%s' % filepath_distill_model)



        input_shape = (maxlen, teacher_w2vdim + 1)
        model_input = Input(shape=input_shape)
        student_model, y_gold = StudentCommon(model_input, teacher_w2vdim, projection_model, maxlen,
                                              pretrained_projection_layer=pretrained_projection_layer)

        student_model.compile(loss=custom_loss_Ba, optimizer="adam")
        data = tuple((x_trn_compound, y_trn, x_dev_compound, y_dev, x_tst_compound, y_tst))
        checkpoint = MyCallback(filepath_distill_model, data, real_save=real_save, monitor='val_loss', verbose=1, save_best_only=True,
                                mode='auto')

    elif loss_name == 'bharat':
        print('bha3')
        if real_save is True:
            model_directory = '../model_distill_sau'
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

        filepath_distill_model = '%s/%s-%s-%s-%d-%f-%d-%d-%d' % \
                                 (model_directory, dataset_name, model_name, projection_model, pretrained, std, teacher_w2vdim,
                                  student_w2vdim, attempt)
        print('sau distill_path=%s' % filepath_distill_model)

        input_shape = (maxlen, teacher_w2vdim + 1)
        model_input = Input(shape=input_shape)
        student_model, y_gold = StudentCommon(model_input, teacher_w2vdim, projection_model, maxlen,
                                              pretrained_projection_layer=pretrained_projection_layer)
        student_model.compile(loss=custom_loss_Bharat(std=std, num_class=num_class), optimizer="adam", metrics=["mse"])
        data = tuple((x_trn_compound, y_trn, x_dev_compound, y_dev, x_tst_compound, y_tst))
        checkpoint = MyCallback(filepath_distill_model, data, real_save=real_save, monitor='val_loss', verbose=1, save_best_only=True,
                                mode='auto')

    else:
        print('mou4')
        if real_save is True:
            model_directory = '../model_distill_mou'
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

        filepath_distill_model = '%s/%s-%s-%s-%d-%d-%d-%d' % \
                                 (model_directory,
                                 dataset_name, model_name, projection_model, pretrained, teacher_w2vdim, student_w2vdim,
                                 attempt)

        print('mou distill_path=%s' % filepath_distill_model)

        input_shape = (maxlen, teacher_w2vdim + 1)
        model_input = Input(shape=input_shape)
        student_model, y_gold = StudentCommon(model_input, teacher_w2vdim, 'vanila', maxlen)
        student_model.compile(loss=custom_loss_Mou(y_gold=y_gold, num_class=num_class), optimizer="adam", metrics=["mse"])
        data = tuple((x_trn_compound, y_trn, x_dev_compound, y_dev, x_tst_compound, y_tst))
        checkpoint = MyCallback(filepath_distill_model, data, real_save=real_save, monitor='val_loss', verbose=1, save_best_only=True,
                                mode='auto')

    print(student_model.count_params())
    print(student_model.summary())
    # exit()

    callbacks_list = [checkpoint]

    # if loss_name == 'bharat':
    student_model.fit(x_trn_compound, y_trn_logit,
                      batch_size=batch_size,
                      callbacks=callbacks_list,
                      epochs=epochs,
                      validation_data=(x_dev_compound, y_dev_logit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default=0, choices=range(10), type=int)
    parser.add_argument('-g', default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"], type=str)
    parser.add_argument('-l', default="ba", choices=["mou", "hinton", "ba", "bharat"], type=str)
    parser.add_argument('-p', default="deep", choices=["vanila", "deep", "conv", "rel"], type=str)
    parser.add_argument('-pt', default=1, choices=[0, 1], type=int) # pre-trained
    parser.add_argument('-s', default=1, choices=[0, 1], type=int) # save
    parser.add_argument('-tau', default=4, type=int)
    parser.add_argument('-a', default=0.3, type=float)
    parser.add_argument('-std', default=0.1, type=float)
    parser.add_argument('-ds', default="sst5", choices=["mr", "sstb", "sst5", "subj", "trec", "cr", "mpqa", "s17"],
                        type=str)
    parser.add_argument('-m', default="cnn2", choices=["cnn1", "cnn2", "att", "lstm"], type=str)
    parser.add_argument('-sttd', default=50, choices=[50, 40, 30, 20, 10], type=int)  # save

    args = parser.parse_args()

    run(args.t, args.g, args.l, bool(args.s), args.tau, args.a, args.std, args.p, bool(args.pt), args.ds, args.m, args.sttd)
