from keras.layers import Input, Dense, Bidirectional, SpatialDropout1D, CuDNNGRU, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Nadam

from src.models.TextModel import ConcPoolModel

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 12
LEARN_RATE = 0.0005
CLIP_NORM = 1.0
NUM_CLASSES = 12


class DoubleBiGRUConcPool(ConcPoolModel):
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        rnn_input = Input(shape=(input_length,))
        embedding = self.embedding_layers(rnn_input, vocab_size, embedding_matrix,
                                          dropout=0.5, noise=0.2,
                                          input_length=input_length, embed_dim=embed_dim)

        bi_gru_1 = Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=l2(0.0001)))(embedding)

        bi_gru_1 = SpatialDropout1D(0.5)(bi_gru_1)

        bi_gru_2, last_state_forward, last_state_back = Bidirectional(CuDNNGRU(64,
                                                                               return_sequences=True,
                                                                               return_state=True,
                                                                               recurrent_regularizer=l2(0.0001)))(bi_gru_1)

        bi_gru_2 = SpatialDropout1D(0.5)(bi_gru_2)

        conc = self.bi_concatenate_pool(bi_gru_2, last_state_forward, last_state_back)

        drop_1 = Dropout(0.5)(conc)

        outputs = Dense(self.num_classes, activation='sigmoid')(drop_1)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)
        model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=self.LEARN_RATE), metrics=['accuracy'])

        if summary:
            model.summary()

        return model
