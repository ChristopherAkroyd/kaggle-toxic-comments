import keras.backend as K

# Only use the amount of memory we require rather than the maximum
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Utility code.
from src.callbacks import RocAucEvaluation
from src.load_data import load_data_split, load_test_data, load_sample_submission
from src.load_glove_embeddings import load_embedding_matrix
from src.write_results import write_results
from src.util import get_save_path
# Model definition
from src.models.bidirectional_GRU_conc_pool import BidirectionalGRUConcPool
from src.layers.Attention import FeedForwardAttention

TRAIN = True
WRITE_RESULTS = True
MAX_FEATS = 200000
SEQUENCE_LENGTH = 150


# Paths to data sets
train_path = './data/train.csv'
test_path = './data/test.csv'
submission_path = './data/sample_submission.csv'
# Paths to glove embeddings.
glove_path = './data/embeddings/glove.42B.300d.txt'
glove_embed_dims = 300


(x_train, y_train), (x_val, y_val), word_index, num_classes, tokenizer = load_data_split(path=train_path, max_features=MAX_FEATS, sequence_length=SEQUENCE_LENGTH)

embedding_matrix = load_embedding_matrix(glove_path=glove_path,
                                         word_index=word_index,
                                         embedding_dimensions=glove_embed_dims)

vocab_size = len(word_index) + 1

model_instance = BidirectionalGRUConcPool(num_classes=num_classes)

print('Number of Data Samples:' + str(len(x_train) + len(x_val)))
print('Number of Classes: ' + str(num_classes))


if TRAIN:
    model = model_instance.build(vocab_size,
                                 embedding_matrix,
                                 input_length=x_train.shape[1],
                                 embed_dim=glove_embed_dims)

    checkpoint = ModelCheckpoint(get_save_path(model_instance), save_best_only=True)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=4,
                               verbose=1,
                               min_delta=0.00001)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2,
                                  verbose=1,
                                  epsilon=0.0001,
                                  mode='min', min_lr=0.0001)

    roc_auc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)

    model.fit(x=x_train,
              y=y_train,
              validation_data=(x_val, y_val),
              epochs=model_instance.EPOCHS,
              batch_size=model_instance.BATCH_SIZE,
              callbacks=[checkpoint, early_stop, roc_auc])

if WRITE_RESULTS:
    test_set = load_test_data(test_path, tokenizer, sequence_length=SEQUENCE_LENGTH)

    submission = load_sample_submission(submission_path)
    write_results(model_instance, test_set, submission,
                  folds=False, custom_objects={'FeedForwardAttention': FeedForwardAttention})
