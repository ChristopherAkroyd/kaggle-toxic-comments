from keras.layers import Input, Dense, SpatialDropout1D, CuDNNGRU
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

from src.models.TextModel import ConcPoolModel

# HPARAMs
BATCH_SIZE = 32
EPOCHS = 15
LEARN_RATE = 0.001
CLIP_NORM = 1.0
NUM_CLASSES = 12


class GRUConcPool(ConcPoolModel):
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

        bi_gru_1, last_state = CuDNNGRU(128, return_sequences=True, return_state=True,
                                        recurrent_regularizer=l2(0.0001),
                                        kernel_regularizer=l2(0.0001))(embedding)

        spatial_dropout_2 = SpatialDropout1D(0.5)(bi_gru_1)

        conc = self.concatenate_pool(spatial_dropout_2, last_state)

        outputs = Dense(self.num_classes, activation='sigmoid')(conc)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.LEARN_RATE, clipnorm=CLIP_NORM),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model
