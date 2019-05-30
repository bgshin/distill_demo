from keras.layers import Input, Dense, Reshape

from .autoencoder_base import AutoencoderBase

class RelationalAutoencoder(AutoencoderBase):
    def __init__(self, h, w, gdims, fdims, projection=True, f_average=True):
        input_dim = h * w
        model_input = Input(shape=(input_dim,), name='%sInput' % 'RN')
        encoded = model_input

        if projection is True:
            encoded = Dense(input_dim, activation='relu', name='Projection')(encoded)

        encoded = self.get_relational_encoder(encoded, h, w, gdims, fdims)
        decoded = self.get_relational_decoder(encoded, h, w, gdims, fdims, f_average=f_average)

        if projection is True:
            decoded = Dense(input_dim, name='DeProjection')(decoded)

        self.compile(model_input, decoded)


class RelationalEncoderConvolutionalDecoder(AutoencoderBase):
    def __init__(self, h, w, gdims, fdims, filters, projection=True):
        input_dim = h * w
        encoded_dim = fdims[-1]
        model_input = Input(shape=(input_dim,), name='%sInput' % 'RN')
        encoded = model_input

        if projection is True:
            encoded = Dense(input_dim, activation='relu', name='Projection')(encoded)

        encoded = self.get_relational_encoder(encoded, h, w, gdims, fdims)

        decoded = Reshape((5, 5, 2), input_shape=(encoded_dim,))(encoded)
        decoded = self.get_conv_decoder(decoded, filters)
        decoded = Reshape((input_dim,), input_shape=(h, w, 1))(decoded)

        if projection is True:
            decoded = Dense(input_dim, name='DeProjection')(decoded)

        self.compile(model_input, decoded)


class RelationalEncoderDeepDecoder(AutoencoderBase):
    def __init__(self, h, w, gdims, fdims, dims, projection=True):
        input_dim = h * w
        encoded_dim = fdims[-1]
        model_input = Input(shape=(input_dim,), name='%sInput' % 'RN')
        encoded = model_input

        if projection is True:
            encoded = Dense(input_dim, activation='relu', name='Projection')(encoded)

        encoded = self.get_relational_encoder(encoded, h, w, gdims, fdims)

        decoded = self.get_deep_decoder(encoded, dims)

        if projection is True:
            decoded = Dense(input_dim, name='DeProjection')(decoded)

        self.compile(model_input, decoded)
