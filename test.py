import keras.backend as K

# Only use the amount of memory we require rather than the maximum
if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

from src.load_data import load_data_folds, load_test_data, load_sample_submission
from src.write_results import power_scale
from keras.models import load_model

MAX_FEATS = 200000
SEQUENCE_LENGTH = 200
EMBEDDINGS = 'GLOVE'
NUM_CLASSES = 6
PATH = './model_checkpoints/BidirectionalGRUConcPool/BidirectionalGRUConcPool_0.0386.hdf5'

# Paths to data sets
train_path = './data/train.csv'
test_path = './data/test.csv'
submission_path = './data/sample_submission.csv'

(x_train, y_train), folds, tokenizer = load_data_folds(path=train_path,
                                                       folds=10,
                                                       max_features=MAX_FEATS,
                                                       sequence_length=SEQUENCE_LENGTH)

test_set = load_test_data(test_path, tokenizer, sequence_length=SEQUENCE_LENGTH)

submission = load_sample_submission(submission_path)

print('Starting to write results...')

print('Running ' + str(len(test_set)) + ' predictions...')

model = load_model(PATH)
predictions = model.predict(test_set)

assert len(predictions) == len(test_set)

df_submission = power_scale(submission)

print('Writing ' + str(len(predictions)) + ' predictions...')

df_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = predictions

df_submission.to_csv('test_submission.csv', index=False)
