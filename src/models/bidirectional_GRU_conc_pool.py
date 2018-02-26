from keras.layers import Input, Dense, Embedding, Bidirectional, SpatialDropout1D, \
    GaussianNoise, CuDNNGRU, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Nadam

# HPARAMs
BATCH_SIZE = 128
EPOCHS = 5
LEARN_RATE = 0.0005
CLIP_NORM = 1.0
NUM_CLASSES = 12


class BidirectionalGRUConcPool:
    def __init__(self, num_classes=NUM_CLASSES):
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.LEARN_RATE = LEARN_RATE
        self.num_classes = num_classes
        self.checkpoint_path = './model_checkpoints/BidirectionalGRUConcPool.hdf5'

    def create_model(self, vocab_size, embedding_matrix, input_length=5000, embed_dim=200):
        input = Input(shape=(input_length, ))

        embedding = Embedding(vocab_size, embed_dim, weights=[embedding_matrix], input_length=input_length)(input)

        spatial_dropout_1 = SpatialDropout1D(0.5)(embedding)

        noise = GaussianNoise(0.2)(spatial_dropout_1)
        bi_gru_1, last_state_forward, last_state_back = Bidirectional(CuDNNGRU(128, return_sequences=True,
                                                                               return_state=True,
                                                                               recurrent_regularizer=l2(0.0001)))(noise)

        bi_gru_1 = SpatialDropout1D(0.5)(bi_gru_1)

        last_state = concatenate([last_state_forward, last_state_back], name='last_state')
        avg_pool = GlobalAveragePooling1D()(bi_gru_1)
        max_pool = GlobalMaxPooling1D()(bi_gru_1)

        conc = concatenate([last_state, max_pool, avg_pool], name='conc_pool')

        drop_1 = Dropout(0.5)(conc)

        outputs = Dense(self.num_classes, activation='sigmoid')(drop_1)

        model = Model(inputs=input, outputs=outputs)

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=self.LEARN_RATE), metrics=['accuracy'])

        return model