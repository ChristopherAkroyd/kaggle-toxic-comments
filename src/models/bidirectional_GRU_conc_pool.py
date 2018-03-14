from keras.layers import Bidirectional, CuDNNGRU, Dense,  SpatialDropout1D, Input, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Nadam

from src.models.TextModel import ConcPoolModel

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.001
# LEARN_RATE = 0.00025
NUM_CLASSES = 12


class BidirectionalGRUConcPool(ConcPoolModel):
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

        bi_gru_1, last_state_forward, last_state_back = Bidirectional(CuDNNGRU(512, return_sequences=True,
                                                                               return_state=True,
                                                                               recurrent_regularizer=l2(0.001),
                                                                               kernel_regularizer=l2(0.001)))(embedding)

        bi_gru_1 = SpatialDropout1D(0.2)(bi_gru_1)

        conc = self.bi_concatenate_pool(bi_gru_1, last_state_forward, last_state_back)

        outputs = Dense(self.num_classes, activation='sigmoid')(conc)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy',
                      optimizer=Nadam(lr=self.LEARN_RATE, schedule_decay=1e-6),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model
