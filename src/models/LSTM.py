from keras.layers import Input, Dense, CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, Dropout, SpatialDropout1D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

from src.models.TextModel import TextModel

# HPARAMs
BATCH_SIZE = 32
EPOCHS = 5
LEARN_RATE = 0.001
NUM_CLASSES = 6


class LSTM(TextModel):
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

        bi_gru_1 = Bidirectional(CuDNNLSTM(50, return_sequences=True,
                                      recurrent_regularizer=l2(0.0001),
                                      kernel_regularizer=l2(0.0001)))(embedding)

        bi_gru_1 = SpatialDropout1D(0.1)(bi_gru_1)

        max_pool = GlobalMaxPooling1D()(bi_gru_1)
        dense_1 = Dense(50, activation="relu")(max_pool)
        drop_1 = Dropout(0.1)(dense_1)

        outputs = Dense(self.num_classes, activation='sigmoid')(drop_1)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.LEARN_RATE),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model
