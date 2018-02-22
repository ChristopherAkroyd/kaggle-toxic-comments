from keras.layers import Input, Dense, Embedding, Bidirectional, BatchNormalization, SpatialDropout1D, GaussianNoise, CuDNNGRU, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

# HPARAMs
BATCH_SIZE = 512
EPOCHS = 5
LEARN_RATE = 0.001
CLIP_NORM = 1.0
NUM_CLASSES = 12


class BidirectionalGRU:
    def __init__(self, num_classes=5):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes
        self.checkpoint_path = './model_checkpoints/Pos_Neg_Classifier.hdf5'

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        input = Input(shape=(input_length, ))

        embedding = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(input)

        spatial_dropout_1 = SpatialDropout1D(0.3)(embedding)

        noise = GaussianNoise(0.2)(spatial_dropout_1)

        batch_norm = BatchNormalization()(noise)

        bi_gru_1 = Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=l2(0.0001)))(batch_norm)
        spatial_dropout_2 = SpatialDropout1D(0.5)(bi_gru_1)

        bi_gru_2 = Bidirectional(CuDNNGRU(64, return_sequences=True, recurrent_regularizer=l2(0.0001)))(spatial_dropout_2)
        spatial_dropout_3 = SpatialDropout1D(0.5)(bi_gru_2)

        avg_pool = GlobalAveragePooling1D()(spatial_dropout_3)
        max_pool = GlobalMaxPooling1D()(spatial_dropout_3)
        conc = concatenate([avg_pool, max_pool])

        outputs = Dense(self.num_classes, activation='sigmoid')(conc)

        model = Model(inputs=input, outputs=outputs)

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.LEARN_RATE, clipnorm=CLIP_NORM), metrics=['accuracy'])

        return model