from keras.layers import Input, Dense, concatenate, MaxPool1D, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import Adam

from src.models.TextModel import CNNModel

# HPARAMs
BATCH_SIZE = 32
EPOCHS = 50
LEARN_RATE = 0.00025
NUM_CLASSES = 12


class TextCNN(CNNModel):
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

        conv_2 = self.convolutional_block(embedding, 128, 2, batch_norm=True, reg=0.0001)
        max_2 = MaxPool1D()(conv_2)
        max_2 = GlobalMaxPool1D()(max_2)

        conv_3 = self.convolutional_block(embedding, 128, 3, batch_norm=True, reg=0.0001)
        max_3 = MaxPool1D()(conv_3)
        max_3 = GlobalMaxPool1D()(max_3)

        conv_4 = self.convolutional_block(embedding, 128, 4, batch_norm=True, reg=0.0001)
        max_4 = MaxPool1D()(conv_4)
        max_4 = GlobalMaxPool1D()(max_4)

        conv_5 = self.convolutional_block(embedding, 128, 5, batch_norm=True, reg=0.0001)
        max_5 = MaxPool1D()(conv_5)
        max_5 = GlobalMaxPool1D()(max_5)

        conc = concatenate([max_2, max_3, max_4, max_5])

        outputs = Dense(self.num_classes, activation='sigmoid')(conc)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.LEARN_RATE), metrics=['accuracy'])

        if summary:
            model.summary()

        return model
