import os
import argparse
from keras import backend as K
from keras.layers import Conv1D, AveragePooling1D, Lambda, Average, Multiply
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding, LSTM
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from distill_datasets import load_all, Timer, maxlen_list
from keras_utils import AccCallback

__author__ = 'Bonggun Shin'


def run(w2vdim, attempt, gpunum, dataset_name, model_name):
    filter_sizes = (2, 3, 4, 5)
    num_filters = 32
    dropout_prob = 0.8
    hidden_dims = 50
    batch_size = 32
    if dataset_name in ['sst5', 'subj', 'mpqa']:
        epochs = 30
    else:
        epochs = 100

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum

    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())


    def CNNv1(model_input, max_features, embedding_matrix, n_class):
        z = Embedding(max_features,
                      w2vdim,
                      weights=[embedding_matrix],
                      input_length=maxlen,
                      trainable=False)(model_input)


        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(z)
            print(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            print(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(n_class, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model


    # input is already embedded
    def CNNv2(model_input, n_class):
        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            print(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            # conv = MaxPooling1D(pool_size=maxlen - sz + 1)(conv)
            print(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        # z = Dropout(dropout_prob)(z)
        model_output = Dense(n_class, activation="softmax")(z)
        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model


    # input is already embedded
    def LSTMv1(model_input, n_class):
        z = Bidirectional(LSTM(units=50,
                               return_sequences=True,
                               dropout=0.2,
                               recurrent_dropout=0.2))(model_input)

        z = Bidirectional(LSTM(units=50,
                               return_sequences=False,
                               dropout=0.2,
                               recurrent_dropout=0.2))(z)

        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(n_class, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model


    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features, maxlen = \
            load_all(w2vdim, dataset_name=dataset_name)

        print ("==================================== [%s]maxlen=%d ====================================" \
              % (dataset_name, maxlen))

    n_class = max(y_trn)+1
    with Timer("Build model..."):
        if model_name=="cnn1":
            input_shape = (maxlen,)
            model_input = Input(shape=input_shape)
            model = CNNv1(model_input, max_features, embedding, n_class)

        elif model_name == "cnn2":
            x_trn = embedding[x_trn]
            x_dev = embedding[x_dev]
            x_tst = embedding[x_tst]
            input_shape = (maxlen, w2vdim)
            model_input = Input(shape=input_shape)
            model = CNNv2(model_input, n_class)

        elif model_name =="lstm":
            batch_size = 300
            x_trn = embedding[x_trn]
            x_dev = embedding[x_dev]
            x_tst = embedding[x_tst]
            input_shape = (maxlen, w2vdim)
            model_input = Input(shape=input_shape)
            model = LSTMv1(model_input, n_class)

        else: # model_name=="lstm"
            x_trn = embedding[x_trn]
            x_dev = embedding[x_dev]
            x_tst = embedding[x_tst]
            input_shape = (maxlen, w2vdim)
            model_input = Input(shape=input_shape)
            model = LSTMv1(model_input, n_class)

        model.summary()

    # checkpoint
    model_directory = '../model'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    filepath='%s/teacher-%s-%s-%d-%d' % (model_directory, dataset_name, model_name, w2vdim, attempt)

    data = tuple((x_trn, y_trn,
                  x_dev, y_dev,
                  x_tst, y_tst
                  ))
    checkpoint = AccCallback(filepath, data=data)
    callbacks_list = [checkpoint]

    with Timer("Training_teacher..."):
        model.fit(x_trn, y_trn,
                  batch_size=batch_size,
                  shuffle=True,
                  callbacks=callbacks_list,
                  epochs=epochs,
                  validation_data=(x_dev, y_dev))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=400, choices=[400], type=int) # embedding dimension
    parser.add_argument('-t', default=0, choices=range(10), type=int) # experiment number (10 teachers)
    parser.add_argument('-g', default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"], type=str) # gpu number
    parser.add_argument('-ds', default="sst5",
                        choices=["mr", "sstb", "sst5", "subj", "trec", "cr", "mpqa"], type=str) # dataset name
    parser.add_argument('-m', default="cnn2", choices=["cnn1", "cnn2", "lstm"], type=str) # model name
    args = parser.parse_args()
    print(args)

    run(args.d, args.t, args.g, args.ds, args.m)

