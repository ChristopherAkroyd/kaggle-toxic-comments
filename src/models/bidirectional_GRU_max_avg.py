from keras.layers import Input, Dense, Bidirectional, SpatialDropout1D, CuDNNGRU
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Nadam

from src.models.TextModel import MaxAvgPoolModel

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
# LEARN_RATE = 0.001
LEARN_RATE = 0.00025
NUM_CLASSES = 12


class BidirectionalGRUMaxAvg(MaxAvgPoolModel):
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        rnn_input = Input(shape=(input_length,))
        embedding = self.embedding_layers(rnn_input, vocab_size, embedding_matrix,
                                          dropout=0.7, noise=0.0,
                                          input_length=input_length, embed_dim=embed_dim)

        bi_gru_1 = Bidirectional(CuDNNGRU(512, return_sequences=True, recurrent_regularizer=l2(0.001),
                                          kernel_regularizer=l2(0.001)))(embedding)

        bi_gru_1 = SpatialDropout1D(0.5)(bi_gru_1)

        conc = self.max_avg_pool(bi_gru_1)

        outputs = Dense(self.num_classes, activation='sigmoid')(conc)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy',
                      optimizer=Nadam(lr=self.LEARN_RATE),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model
