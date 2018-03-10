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
from src.load_data import load_data_split, load_data_folds, load_test_data, load_sample_submission
from src.load_embeddings import load_embeddings
from src.write_results import write_results
from src.util import get_save_path
# Model definition
from src.models.bidirectional_GRU_conc_pool import BidirectionalGRUConcPool
# Custom Layers
from src.layers.Attention import FeedForwardAttention

TRAIN = True
WRITE_RESULTS = True
MAX_FEATS = 200000
SEQUENCE_LENGTH = 200
EMBEDDINGS = 'GLOVE'
NUM_CLASSES = 6
FOLDS = -1

# Paths to data sets
train_path = './data/train.csv'
test_path = './data/test.csv'
submission_path = './data/sample_submission.csv'
# Paths to glove embeddings.
glove_path = './data/embeddings/glove.42B.300d.txt'
# glove_path = './data/embeddings/glove.840B.300d.txt'
fast_text_path = './data/embeddings/crawl-300d-2M.vec'
embedding_dimension = 300

if EMBEDDINGS == 'GLOVE':
    embedding_path = glove_path
elif EMBEDDINGS == 'FAST_TEXT':
    embedding_path = fast_text_path
else:
    embedding_path = ''

if FOLDS > 0:
    (x_train, y_train), folds, tokenizer = load_data_folds(path=train_path,
                                                           folds=FOLDS,
                                                           max_features=MAX_FEATS,
                                                           sequence_length=SEQUENCE_LENGTH)
else:
    (x_train, y_train), (x_val, y_val), tokenizer = load_data_split(path=train_path,
                                                                    max_features=MAX_FEATS,
                                                                    sequence_length=SEQUENCE_LENGTH)

embedding_matrix = load_embeddings(path=embedding_path,
                                   embedding_type=EMBEDDINGS,
                                   word_index=tokenizer.word_index,
                                   max_features=MAX_FEATS,
                                   embedding_dimensions=embedding_dimension)

vocab_size = len(tokenizer.word_index) + 1

model_instance = BidirectionalGRUConcPool(num_classes=NUM_CLASSES)

print('Number of Data Samples:' + str(len(x_train) + len(x_val)))
print('Number of Classes: ' + str(NUM_CLASSES))

if TRAIN:
    model = model_instance.build(vocab_size,
                                 embedding_matrix,
                                 input_length=x_train.shape[1],
                                 embed_dim=embedding_dimension,
                                 summary=False)

    checkpoint = ModelCheckpoint(get_save_path(model_instance), save_best_only=True)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1,
                               min_delta=0.00001)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=2,
                                  verbose=1,
                                  epsilon=0.0001,
                                  mode='min', min_lr=0.0001)

    if FOLDS > 0:
        # Store initial weights
        init_weights = model.get_weights()

        print('Running {}-Fold Cross Validation..'.format(FOLDS))

        for i, (train, test) in enumerate(folds):
            print('Fold:' + str(i + 1))
            f_x_train, f_y_train = x_train[train], y_train[train]
            x_val, y_val = x_train[test], y_train[test]

            roc_auc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
            checkpoint = ModelCheckpoint(get_save_path(model_instance, fold=i), save_best_only=True)

            model.fit(x=f_x_train,
                      y=f_y_train,
                      validation_data=(x_val, y_val),
                      epochs=model_instance.EPOCHS,
                      batch_size=model_instance.BATCH_SIZE,
                      callbacks=[early_stop, roc_auc, checkpoint])

            model.set_weights(init_weights)

        model = None
        K.clear_session()
    else:
        roc_auc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)

        print('Training Model...')

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
                  folds=FOLDS, custom_objects={'FeedForwardAttention': FeedForwardAttention})
