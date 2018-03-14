from keras.layers import Input, Dense, CuDNNGRU, Bidirectional, GlobalMaxPooling1D, Dropout, SpatialDropout1D, Reshape
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

from src.layers.Attention import FeedForwardAttention
from src.models.TextModel import TextModel

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 5
LEARN_RATE = 0.001
NUM_CLASSES = 6


class HAN(TextModel):
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        rnn_input = Input(shape=(input_length,))
        embedding = self.embedding_layers(rnn_input, vocab_size, embedding_matrix,
                                          dropout=0.5, noise=0.0,
                                          input_length=input_length, embed_dim=embed_dim)

        word_encoder = Bidirectional(CuDNNGRU(50, return_sequences=True,
                                              recurrent_regularizer=l2(0.0001),
                                              kernel_regularizer=l2(0.0001)))(embedding)

        word_encoder = SpatialDropout1D(0.1)(word_encoder)

        word_attention = FeedForwardAttention()(word_encoder)

        word_attention = Dropout(0.5)(word_attention)

        word_attention = Reshape((1, 50 * 2))(word_attention)

        sentence_encoder = Bidirectional(CuDNNGRU(50, return_sequences=True,
                                         recurrent_regularizer=l2(0.0001),
                                         kernel_regularizer=l2(0.0001)))(word_attention)

        sentence_encoder = SpatialDropout1D(0.1)(sentence_encoder)

        sent_attention = FeedForwardAttention()(sentence_encoder)

        sent_attention = Dropout(0.5)(sent_attention)

        outputs = Dense(self.num_classes, activation='sigmoid')(sent_attention)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.LEARN_RATE),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model
