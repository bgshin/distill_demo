import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import ModelCheckpoint
from distill_datasets import load_all, Timer
import os
from sklearn.model_selection import train_test_split
import gc
import argparse
from src.autoencoder import *
from distill_datasets import get_embedding
import sys

__author__ = 'Bonggun Shin'


def get_dataset(dataset_name):
    with Timer("load_all..."):
        embedding, vocab = get_embedding(400, dataset_name=dataset_name)
        x_trn, x_tst, y_trn, y_tst = train_test_split(embedding, embedding, test_size=0.3, random_state=42)

        del (y_trn)
        del (y_tst)
        gc.collect()

    return x_trn, x_tst


def run(x_trn, x_tst, gpu_num, encoding_dim, model_name, dataset_name):
    # set parameters:
    embedding_dims = 400
    # encoding_dim = 50
    maxlen = 60
    epochs = 100

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())

    model_directory = '../model_ae'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    with Timer("Build model..."):
        if model_name=='v':
            autoenc = Autoencoder(embedding_dims, encoding_dim)
            filepath = '%s/selu_vanila-%s-%d-%d' % (model_directory, dataset_name, embedding_dims, encoding_dim)

        elif model_name=='d':
            autoenc = DeepAutoencoder([embedding_dims, 400, encoding_dim])
            filepath = '%s/selu_deep-%s-%d-%d' % (model_directory, dataset_name, embedding_dims, encoding_dim)

        elif model_name == 'vd':
            autoenc = DeepAutoencoder([embedding_dims, 2000, 400, encoding_dim])
            filepath = '%s/vdeep-new' % (model_directory)

        elif model_name == 'vd2':
            autoenc = DeepAutoencoder([embedding_dims, 400, 200, 100, encoding_dim])
            filepath = '%s/vdeep2-new' % (model_directory)

        elif model_name == 'vd3':
            autoenc = DeepAutoencoder([embedding_dims, 5000, encoding_dim])
            filepath = '%s/vdeep3-new' % (model_directory)

        elif model_name=='c':
            autoenc = ConvolutionalAutoencoder(20, 20, [(8, 3, 3), (2, 3, 3)])
            filepath = '%s/conv-new' % (model_directory)

        elif model_name=='r':
            autoenc = RelationalAutoencoder(100, 4, [100, 50], [50, 50])
            filepath = '%s/rel-new' % (model_directory)

        elif model_name=='rc':
            autoenc = RelationalEncoderConvolutionalDecoder(100, 4, [100, 50], [50, 50], [(8, 3, 3), (2, 3, 3)])
            filepath = '%s/relconv-new' % (model_directory)

        elif model_name=='rd':
            autoenc = RelationalEncoderDeepDecoder(100, 4, [100, 50], [50, 50], [400, 100, 0])
            filepath = '%s/reldeep-new' % (model_directory)

        else:
            print('wrong model name (%s)' % model_name)
            exit()

        autoenc.summary()


    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                                 save_weights_only=True)
    callbacks_list = [checkpoint]
    autoenc.train(x_trn, x_tst, epochs, 2000, callbacks=callbacks_list)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default="0", choices=["0", "1", "2", "3"], type=str)
    parser.add_argument('-d', default=50, choices=[50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], type=int)
    parser.add_argument('-m', default="d", choices=["v", "d", "vd", "vd2", "vd3", "c", "r", "rc", "rd"], type=str)
    parser.add_argument('-emb', default="sst5", choices=["sst5", "news"], type=str)
    args = parser.parse_args()
    print(args)

    x_trn, x_tst = get_dataset(args.emb)
    # one at a time mode
    # run(x_trn, x_tst, args.g, args.d, args.m, args.emb)

    # batch mode
    # for sdim in range(9,0,-1):
    for sdim in range(50, 0, -10):
        for model_name in ["v", "d"]:
            if model_name is "v":
                print('================================ [%s] vanila sdim=%d ================================ ' % (args.emb, sdim))
            else:
                print('================================ [%s] deep sdim=%d ================================ ' % (args.emb, sdim))
            sys.stdout.flush()

            run(x_trn, x_tst, args.g, sdim, model_name, args.emb)






