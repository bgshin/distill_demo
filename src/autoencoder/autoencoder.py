from keras.layers import Input, Dense
from keras.models import Model

from .autoencoder_base import AutoencoderBase


class Autoencoder(AutoencoderBase):
    def __init__(self, input_dim, encoded_dim):
        model_input = Input(shape=(input_dim,), name='%sInput' % 'Deep')
        encoded = self.get_vanila_encoder(model_input, input_dim, encoded_dim)
        decoded = self.get_vanila_decoder(encoded, input_dim, encoded_dim)
        self.compile(model_input, decoded)