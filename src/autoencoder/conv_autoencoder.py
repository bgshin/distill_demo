from .autoencoder_base import AutoencoderBase
from keras.layers import Input, Dense, Reshape

class ConvolutionalAutoencoder(AutoencoderBase):

    def __init__(self, nrow, ncol, filters, projection=True):
        input_dim = nrow*ncol
        model_input = Input(shape=(input_dim,), name='ConvEncoderIn')
        encoded = model_input

        if projection is True:
            encoded = Dense(input_dim, activation='relu', name='Projection')(encoded)

        encoded = Reshape((nrow, ncol, 1), input_shape=(input_dim,))(encoded)
        encoded = self.get_conv_encoder(encoded, filters)

        decoded = self.get_conv_decoder(encoded, filters)
        decoded = Reshape((input_dim,), input_shape=(nrow, ncol, 1))(decoded)

        if projection is True:
            decoded = Dense(input_dim, name='DeProjection')(decoded)

        self.compile(model_input, decoded)
