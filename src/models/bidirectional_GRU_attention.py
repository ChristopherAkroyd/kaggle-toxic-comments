from keras.layers import Dense, Embedding, Bidirectional, SpatialDropout1D, GaussianNoise, CuDNNGRU
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Nadam

from src.layers.Attention import FeedForwardAttention as Attention

from src.models.TextModel import TextModel

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 32
LEARN_RATE = 0.0001
NUM_CLASSES = 12


class BidirectionalGRUAttention(TextModel):
    def __init__(self, num_classes=6):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        model = Sequential()

        model.add(Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length))
        model.add(SpatialDropout1D(0.5))
        model.add(GaussianNoise(0.2))

        model.add(Bidirectional(CuDNNGRU(256, return_sequences=True,
                                         recurrent_regularizer=l2(0.0001),
                                         kernel_regularizer=l2(0.0001))))
        model.add(SpatialDropout1D(0.5))

        model.add(Attention())

        model.add(Dense(self.num_classes, activation='sigmoid'))

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=self.LEARN_RATE, clipvalue=1, clipnorm=1),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model
