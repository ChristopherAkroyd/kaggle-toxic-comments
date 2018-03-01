from keras.layers import Input, Dense, Embedding, Bidirectional, SpatialDropout1D, \
    GaussianNoise, CuDNNGRU, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.001
NUM_CLASSES = 12


class BidirectionalGRUMaxAvg:
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        input = Input(shape=(input_length, ))

        embedding = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(input)

        spatial_dropout_1 = SpatialDropout1D(0.5)(embedding)

        noise = GaussianNoise(0.2)(spatial_dropout_1)
        bi_gru_1 = Bidirectional(CuDNNGRU(512, return_sequences=True, recurrent_regularizer=l2(0.0001),
                                          kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)))(noise)

        bi_gru_1 = SpatialDropout1D(0.5)(bi_gru_1)

        avg_pool = GlobalAveragePooling1D()(bi_gru_1)
        max_pool = GlobalMaxPooling1D()(bi_gru_1)

        conc = concatenate([max_pool, avg_pool])

        drop_1 = Dropout(0.3)(conc)

        dense_1 = Dense(256, activation='relu', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001))(drop_1)

        drop_1 = Dropout(0.5)(dense_1)

        outputs = Dense(self.num_classes, activation='sigmoid')(drop_1)

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
