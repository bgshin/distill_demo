from keras import backend as K
from keras.layers import Conv1D, AveragePooling1D, Lambda, Average, Multiply
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Embedding, LSTM
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers.wrappers import Bidirectional


def CNNv1(model_input, module_input, filter_sizes, num_filters, dropout_prob, hidden_dims, dataset_name,
          max_features, embedding_matrix, w2vdim, maxlen, n_class, path_trained_weights=None, activation=None):
    z = Embedding(max_features,
                  w2vdim,
                  weights=[embedding_matrix],
                  input_length=maxlen,
                  trainable=False)(module_input)

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
    # model_output = Dense(5, activation="softmax")(z)

    if activation is None:
        print("[INFO] no activation")
    else:
        print("[INFO] softmax activation")

    model_output = Dense(n_class, activation=activation)(z)

    model = Model(model_input, model_output)
    model.load_weights(path_trained_weights)

    return model


# input is already embedded
def CNNv2(model_input, module_input, filter_sizes, num_filters, dropout_prob, hidden_dims, dataset_name,
          path_trained_weights=None, activation=None):
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(num_filters,
                      sz,
                      padding="valid",
                      activation="relu",
                      strides=1)(module_input)
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

    if activation is None:
        print("[INFO] no activation")
    else:
        print("[INFO] softmax activation")



    if dataset_name == 'sst5':
        model_output = Dense(5, activation=activation)(z)
    elif dataset_name == 'sstb' or dataset_name == 'mpqa' or dataset_name == 'mr' or dataset_name == 'cr' \
            or dataset_name == 'subj':
        model_output = Dense(2, activation=activation)(z)
    elif dataset_name == 'trec':
        model_output = Dense(6, activation=activation)(z)
    elif dataset_name == 's17':
        model_output = Dense(3, activation=activation)(z)
    else:
        model_output = Dense(2, activation=activation)(z)

    model = Model(model_input, model_output)
    # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    if path_trained_weights is not None:
        model.load_weights(path_trained_weights)
    return model


# input is already embedded
def LSTMv1(model_input, module_input, dropout_prob, hidden_dims, dataset_name, path_trained_weights=None, activation=None):
    conv_blocks = []
    # z = Bidirectional(LSTM(units=50,
    #                        return_sequences=True,
    #                        recurrent_dropout=0.5,
    #                        dropout=0.5))(model_input)

    z = Bidirectional(LSTM(units=50,
                           return_sequences=True,
                           dropout=0.2,
                           recurrent_dropout=0.2))(module_input)

    z = Bidirectional(LSTM(units=50,
                           return_sequences=False,
                           dropout=0.2,
                           recurrent_dropout=0.2))(z)

    z = Dropout(dropout_prob)(z)
    z = Dense(hidden_dims, activation="relu")(z)
    # z = Dropout(dropout_prob)(z)
    if dataset_name == 'sst5':
        model_output = Dense(5, activation=activation)(z)
    elif dataset_name == 'sstb' or dataset_name == 'mpqa' or dataset_name == 'mr' or dataset_name == 'cr' \
            or dataset_name == 'subj':
        model_output = Dense(2, activation=activation)(z)
    elif dataset_name == 'trec':
        model_output = Dense(6, activation=activation)(z)
    elif dataset_name == 's17':
        model_output = Dense(3, activation=activation)(z)
    else:
        model_output = Dense(2, activation=activation)(z)

    model = Model(model_input, model_output)
    if path_trained_weights is not None:
        model.load_weights(path_trained_weights)

    # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
