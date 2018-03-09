from keras.layers import Input, Dense, Embedding, SpatialDropout1D, GaussianNoise,\
    GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, BatchNormalization, Activation, add, MaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.001
NUM_CLASSES = 12


class DPCNN:
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        input = Input(shape=(input_length, ))

        embedding = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(input)

        spatial_dropout_1 = SpatialDropout1D(0.5)(embedding)

        conv_block_1 = Conv1D(64, kernel_size=3, padding='same', activation='linear', kernel_reguarlizer=l2(0.001))(spatial_dropout_1)
        conv_block_1 = BatchNormalization()(conv_block_1)
        conv_block_1 = SpatialDropout1D(0.2)(conv_block_1)
        conv_block_1 = Activation('relu')(conv_block_1)

        conv_block_2 = Conv1D(64, kernel_size=3, padding='same', activation='linear', kernel_reguarlizer=l2(0.001))(
            conv_block_1)
        conv_block_2 = BatchNormalization()(conv_block_2)
        conv_block_2 = SpatialDropout1D(0.2)(conv_block_2)
        conv_block_2 = Activation('relu')(conv_block_2)

        shape_conv = Conv1D(64, kernel_size=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(embedding)

        first_stage_out = add([shape_conv, conv_block_2])

        x = MaxPooling1D(pool_size=3, strides=2)(first_stage_out)





        outputs = Dense(self.num_classes, activation='sigmoid')(conc)

        model = Model(inputs=input, outputs=outputs)

        return model

    def build(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200, summary=True):
        model = self.create_model(vocab_size, embedding_matrix, input_length, embed_dim)

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=self.LEARN_RATE),
                      metrics=['accuracy'])

        if summary:
            model.summary()

        return model