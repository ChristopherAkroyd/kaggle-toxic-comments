from keras.layers import concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, Embedding, SpatialDropout1D, \
    GaussianNoise, Conv1D, Activation, BatchNormalization
from keras.regularizers import l2


class TextModel:
    def embedding_layers(self, tensor, vocab_size, embedding_matrix, dropout=0.5, noise=0.2, input_length=5000,
                         embed_dim=200):
        embedding = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(tensor)

        spatial_dropout_1 = SpatialDropout1D(dropout)(embedding)

        noise = GaussianNoise(noise)(spatial_dropout_1)

        return noise

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        pass

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        pass


class ConcPoolModel(TextModel):
    def bi_concatenate_pool(self, rnn, state_forward, state_back):
        state = concatenate([state_forward, state_back], name='last_state')
        return self.concatenate_pool(rnn, state)

    def concatenate_pool(self, rnn, state):
        avg_pool = GlobalAveragePooling1D()(rnn)
        max_pool = GlobalMaxPooling1D()(rnn)

        concatenate_pool = concatenate([state, max_pool, avg_pool], name='concatenate_pool')
        return concatenate_pool


class MaxAvgPoolModel(TextModel):
    def max_avg_pool(self, rnn):
        avg_pool = GlobalAveragePooling1D()(rnn)
        max_pool = GlobalMaxPooling1D()(rnn)

        concatenate_pool = concatenate([max_pool, avg_pool], name='max_avg_pool')
        return concatenate_pool


class CNNModel(TextModel):
    def convolutional_block(self, tensor, filters=128, kernel_size=3, batch_norm=False, dropout=None, reg=0.001):

        conv_block = Conv1D(filters,
                            kernel_size=kernel_size,
                            padding='same',
                            kernel_regularizer=l2(reg),
                            activation='linear')(tensor)

        if batch_norm:
            conv_block = BatchNormalization()(conv_block)

        if dropout:
            conv_block = SpatialDropout1D(dropout)(conv_block)

        conv_block = Activation('relu')(conv_block)

        return conv_block
