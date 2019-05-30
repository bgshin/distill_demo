from keras import backend
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import Reshape, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Concatenate, Add, Average

class AutoencoderBase(object):
    def train(self, x_train, x_test, epochs, batch_size, stop_early=True, callbacks=None):
        if callbacks is None:
            callbacks = []

        if stop_early:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'))

        self.autoencoder.fit(x_train, x_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(x_test, x_test),
                             callbacks=callbacks)

    def encode(self, x):
        return self.encoder.predict(x)

    def decode(self, x):
        return self.decoder.predict(x)

    def summary(self):
        self.autoencoder.summary()

    def get_deep_encoder(self, model_input, dims, name_prefix='Deep'):
        dims_encoder = dims[1:]

        # model_input = Input(shape=(dim_in,), name='%sInput' % name_prefix)

        encoded = model_input
        dense_activation = 'relu'
        # Construct encoder layers
        for i, dim in enumerate(dims_encoder):
            name = '{0}Encoder{1}'.format(name_prefix, i)
            if i == len(dims_encoder) - 1:
                dense_activation = None

            encoded = Dense(dim, activation=dense_activation, name=name)(encoded)

        return encoded

    def get_deep_decoder(self, encoded, dims, activation=None, name_prefix='Deep'):
        dims_decoding = dims[:-1]
        dims_decoding.reverse()

        decoded = encoded
        for i, dim in enumerate(dims_decoding):
            name = '{0}Decoder{1}'.format(name_prefix, i)

            dense_activation = 'relu'
            if i == len(dims_decoding) - 1:
                dense_activation = activation
                name = '{0}Output'.format(name_prefix, i)

            layer = Dense(dim, activation=dense_activation, name=name)

            decoded = layer(decoded)

        return decoded

    def get_vanila_encoder(self, model_input, input_dim, encoded_dim):
        dims = [input_dim, encoded_dim]
        return self.get_deep_encoder(model_input, dims, name_prefix='Vanila')

    def get_vanila_decoder(self, encoded, input_dim, encoded_dim, activation=None):
        dims = [input_dim, encoded_dim]
        return self.get_deep_decoder(encoded, dims, activation=activation, name_prefix='Vanila')

    def get_conv_encoder(self, model_input, filters):
        encoded = model_input

        for i, (n_filter, w, h) in enumerate(filters):
            name = 'Conv{0}'.format(i)
            encoded = Conv2D(n_filter, [w, h], activation='relu', padding='same', name=name)(encoded)
            encoded = MaxPooling2D((2, 2), padding='same', name='MaxPool{0}'.format(i))(encoded)

        return encoded

    def get_conv_decoder(self, encoded, filters, activation=None):
        decoded = encoded
        for i, (n_filter, w, h) in enumerate(reversed(filters)):
            convlayer = Conv2D(n_filter, [w, h], activation='relu', padding='same', name='Deconv{0}'.format(i))
            decoded = convlayer(decoded)

            upsample = UpSampling2D((2, 2), name='UpSampling{0}'.format(i))
            decoded = upsample(decoded)

        # Reduce from X filters to 1 in the output layer. Make sure its sigmoid for the [0..1] range
        convlayer = Conv2D(1, [filters[0][0], filters[0][1]], activation=activation, padding='same')
        decoded = convlayer(decoded)
        return decoded

    def get_relational_encoder(self, model_input, h, w, gdims, fdims):
        input_dim = h * w

        encoded = Reshape((h, w), input_shape=(input_dim,))(model_input)

        g_list = []
        for idx, gdim in enumerate(gdims):
            g = Dense(gdim, activation='relu', name='G%d' % idx)
            g_list.append(g)

        f_list = []
        for idx, fdim in enumerate(fdims):
            f = Dense(fdim, activation='relu', name='F%d' % idx)
            f_list.append(f)

        g_outputs = []
        for i in range(w):
            for j in range(w):
                object_i = Lambda(lambda x: x[:, :, i])(encoded)
                object_j = Lambda(lambda x: x[:, :, j])(encoded)

                g_input = Concatenate(axis=1)([object_i, object_j])
                gz = g_input
                for g in g_list:
                    gz = g(gz)

                g_outputs.append(gz)

        f_input = Add()(g_outputs) if len(g_outputs) > 1 else gz

        fz = f_input
        for f in f_list:
            fz = f(fz)

        encoded = fz
        return encoded

    def get_relational_decoder(self, encoded, h, w, gdims, fdims, f_average=True):
        gdims[0] = h * 2
        g_list = []
        for idx, gdim in reversed(list(enumerate(gdims))):
            g = Dense(gdim, activation='relu', name='deG%d' % idx)
            g_list.append(g)

        f_list = []
        for idx, fdim in reversed(list(enumerate(fdims))):
            f = Dense(fdim, activation='relu', name='deF%d' % idx)
            f_list.append(f)

        fz = encoded
        for f in f_list:
            fz = f(fz)

        gd_outputs = {}
        for i in range(w):
            for j in range(w):
                gz = fz
                for g in g_list:
                    gz = g(gz)

                gd_z_i = Lambda(lambda x: x[:, 0:h])(gz)
                gd_z_j = Lambda(lambda x: x[:, h:])(gz)

                if i in gd_outputs:
                    gd_outputs[i].append(gd_z_i)
                else:
                    gd_outputs[i] = [gd_z_i]

                if j in gd_outputs:
                    gd_outputs[i].append(gd_z_j)
                else:
                    gd_outputs[j] = [gd_z_j]

        gd_segments = []
        for i in gd_outputs.keys():
            if f_average is True:
                added_segment = Average()(gd_outputs[i])
            else:
                added_segment = Add()(gd_outputs[i])
            gd_segments.append(added_segment)

        decoded = Concatenate(axis=1)(gd_segments)

        return decoded

    def compile(self, model_input, decoded, loss='mean_squared_error', optimizer='Nadam'):
        self.autoencoder = Model(model_input, decoded)
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
