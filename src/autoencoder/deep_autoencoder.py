from keras.layers import Input
from .autoencoder_base import AutoencoderBase

class DeepAutoencoder(AutoencoderBase):

    def __init__(self, dims):
        input_dim = dims[0]
        model_input = Input(shape=(input_dim,), name='%sInput' % 'Deep')
        encoded = self.get_deep_encoder(model_input, dims)
        decoded = self.get_deep_decoder(encoded, dims)
        self.compile(model_input, decoded)
