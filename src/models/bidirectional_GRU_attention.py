from keras.layers import Dense, Embedding, Bidirectional, Dropout, BatchNormalization, SpatialDropout1D, GaussianNoise, CuDNNGRU
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam

from src.layers.Attention import FeedForwardAttention as Attention

# HPARAMs
BATCH_SIZE = 512
EPOCHS = 8
LEARN_RATE = 0.001
CLIP_NORM = 1.0
NUM_CLASSES = 12


class BidirectionalGRUAttention:
    def __init__(self, num_classes=6):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        model = Sequential()

        model.add(Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length))
        model.add(SpatialDropout1D(0.3))
        model.add(GaussianNoise(0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=l2(0.0001))))
        model.add(SpatialDropout1D(0.5))
        model.add(Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=l2(0.0001))))
        model.add(SpatialDropout1D(0.5))

        model.add(Attention())
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes, activation='sigmoid'))

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.LEARN_RATE, clipnorm=CLIP_NORM),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model
