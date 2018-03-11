from keras.layers import Input, Dense, GlobalMaxPooling1D, Conv1D, add, MaxPooling1D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

from src.models.TextModel import CNNModel

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.005
NUM_CLASSES = 12
DPCNN_DEPTH = 3


class DPCNN(CNNModel):
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.DPCNN_DEPTH = DPCNN_DEPTH
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        rnn_input = Input(shape=(input_length,))

        embedding = self.embedding_layers(rnn_input, vocab_size, embedding_matrix,
                                          dropout=0.5, noise=0.0,
                                          input_length=input_length, embed_dim=embed_dim)

        conv_block_1 = self.convolutional_block(embedding, filters=64, batch_norm=True, dropout=0.2, reg=0.00001)
        conv_block_2 = self.convolutional_block(conv_block_1, filters=64, batch_norm=True, dropout=0.2, reg=0.00001)

        shape_conv = Conv1D(64, kernel_size=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(embedding)

        first_stage_out = add([shape_conv, conv_block_2])

        dpcnn_stage = first_stage_out

        for i in range(self.DPCNN_DEPTH):
            dpcnn_pool = MaxPooling1D(pool_size=3, strides=2)(dpcnn_stage)
            dpcnn_block = self.convolutional_block(dpcnn_pool, filters=64,
                                                   batch_norm=True, dropout=0.2, reg=0.00001)

            dpcnn_block = self.convolutional_block(dpcnn_block, filters=64,
                                                   batch_norm=True, dropout=0.2, reg=0.00001)

            dpcnn_stage = add([dpcnn_block, dpcnn_pool])

        max_pool = GlobalMaxPooling1D()(dpcnn_stage)

        dense_1 = Dense(256, kernel_regularizer=l2(0.00001))(max_pool)
        drop_1 = Dropout(0.5)(dense_1)

        outputs = Dense(self.num_classes, activation='sigmoid')(drop_1)

        model = Model(inputs=rnn_input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=self.LEARN_RATE),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model