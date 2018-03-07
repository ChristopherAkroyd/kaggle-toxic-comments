from keras.layers import Input, Dense, Embedding, SpatialDropout1D, GaussianNoise, BatchNormalization,\
    Activation, Conv1D, concatenate, MaxPool1D, GlobalMaxPool1D, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 32
EPOCHS = 50
LEARN_RATE = 0.00025
NUM_CLASSES = 12


class TextCNN:
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        input = Input(shape=(input_length, ))

        embedding = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(input)

        spatial_dropout_1 = SpatialDropout1D(0.5)(embedding)

        conv_2 = Conv1D(128, kernel_size=2, padding='same', kernel_regularizer=l2(0.0001))(spatial_dropout_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation('relu')(conv_2)
        max_2 = MaxPool1D()(conv_2)
        max_2 = GlobalMaxPool1D()(max_2)

        conv_3 = Conv1D(128, kernel_size=3, padding='same', kernel_regularizer=l2(0.0001))(spatial_dropout_1)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation('relu')(conv_3)
        max_3 = MaxPool1D()(conv_3)
        max_3 = GlobalMaxPool1D()(max_3)

        conv_4 = Conv1D(128, kernel_size=4, padding='same', kernel_regularizer=l2(0.0001))(spatial_dropout_1)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation('relu')(conv_4)
        max_4 = MaxPool1D()(conv_4)
        max_4 = GlobalMaxPool1D()(max_4)

        conv_5 = Conv1D(128, kernel_size=5, padding='same', kernel_regularizer=l2(0.0001))(spatial_dropout_1)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = Activation('relu')(conv_5)
        max_5 = MaxPool1D()(conv_5)
        max_5 = GlobalMaxPool1D()(max_5)

        conc = concatenate([max_2, max_3, max_4, max_5])

        outputs = Dense(self.num_classes, activation='sigmoid')(conc)

        model = Model(inputs=input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.LEARN_RATE), metrics=['accuracy'])

        if summary:
            model.summary()

        return model
