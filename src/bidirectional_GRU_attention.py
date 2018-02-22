from keras.layers import Dense, Embedding, Bidirectional, Dropout, BatchNormalization, SpatialDropout1D, CuDNNLSTM, GaussianNoise, CuDNNGRU
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop

from .layers.Attention import FeedForwardAttention as Attention

# HPARAMs
BATCH_SIZE = 512
EPOCHS = 5
LEARN_RATE = 0.001
CLIP_NORM = 1.0
NUM_CLASSES = 12


class BidirectionalGRUAttention:
    def __init__(self, num_classes=6):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes
        self.checkpoint_path = './model_checkpoints/Pos_Neg_Classifier.hdf5'

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

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.LEARN_RATE, clipnorm=CLIP_NORM), metrics=['accuracy'])

        return model