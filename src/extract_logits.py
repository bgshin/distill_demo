from keras.layers import Input
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np
import argparse
from distill_datasets import load_all, Timer
from models import CNNv2, LSTMv1
import _pickle as cPickle
from keras import backend as K


def run(gpunum, dataset_name):
    # set parameters:
    teacher_w2vdim = 400
    filter_sizes = (2, 3, 4, 5)
    num_filters = 32
    hidden_dims = 50

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum
    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())



    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features, maxlen = \
            load_all(teacher_w2vdim, dataset_name=dataset_name)

        print("==================================== [%s]maxlen=%d ====================================" \
              % (dataset_name, maxlen))

    x_trn = embedding[x_trn]
    x_dev = embedding[x_dev]
    x_tst = embedding[x_tst]

    logit_dic = {}


    for model_name in ['cnn2', 'lstm']:
        for attempt in range(10):
            input_shape = (maxlen, teacher_w2vdim)
            model_input = Input(shape=input_shape)

            with Timer("Build Teacher Model..."):
                teacher_model_path = '../model/teacher-%s-%s-%d-%d' % (dataset_name, model_name, teacher_w2vdim, attempt)
                print('teacher_model_path=%s' % teacher_model_path)

                if model_name == "cnn2":
                    teacher_model = CNNv2(model_input, model_input, filter_sizes, num_filters, 0.0,
                                          hidden_dims, dataset_name,
                                          path_trained_weights=teacher_model_path, activation=None)

                elif model_name == "lstm":
                    teacher_model = LSTMv1(model_input, model_input, 0.0, hidden_dims, dataset_name,
                                           path_trained_weights=teacher_model_path, activation=None)

            with Timer("Get Teacher's Logits..."):
                y_trn_logit = np.array(teacher_model.predict(x_trn, verbose=1))
                y_dev_logit = np.array(teacher_model.predict(x_dev, verbose=1))
                y_tst_logit = np.array(teacher_model.predict(x_tst, verbose=1))

            data = (y_trn_logit, y_dev_logit, y_tst_logit)
            logit_dic['%s_%d'%(model_name, attempt)] = data

            K.clear_session()

    cPickle.dump(logit_dic, open('../data/logit_dic_%s.pkl' % (dataset_name), 'wb'), protocol=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default="0", choices=["0", "1", "2", "3"], type=str)
    parser.add_argument('-ds', default="sst5", choices=["mr", "sstb", "sst5", "subj", "trec", "cr", "mpqa", "s17"],
                        type=str)

    args = parser.parse_args()

    run(args.g, args.ds)
