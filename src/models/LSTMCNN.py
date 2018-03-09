from keras.layers import Input, Dense, Embedding, Bidirectional, SpatialDropout1D, concatenate,\
    GlobalAveragePooling1D, GlobalMaxPooling1D, CuDNNLSTM, Conv1D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 50
LEARN_RATE = 0.001
NUM_CLASSES = 12


class LSTMCNN:
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        input = Input(shape=(input_length, ))

        embedding = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(input)

        spatial_dropout_1 = SpatialDropout1D(0.2)(embedding)

        bi_lstm = Bidirectional(CuDNNLSTM(128, return_sequences=True))(spatial_dropout_1)

        bi_lstm = SpatialDropout1D(0.5)(bi_lstm)

        conv_lstm = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(bi_lstm)
        conv_lstm = Dropout(0.5)(conv_lstm)

        avg_pool = GlobalAveragePooling1D()(conv_lstm)
        max_pool = GlobalMaxPooling1D()(conv_lstm)

        conc = concatenate([max_pool, avg_pool], name='conc_pool')

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