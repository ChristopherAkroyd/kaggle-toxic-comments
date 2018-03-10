import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold

# Tokenize and sequence padding.
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from src.preprocessing import TextPreProcessor

RANDOM_SEED = 233
MAX_SEQ_LEN = 100
MAX_FEATS = 15000
REMOVE_NUMBERS = False
TEXT_KEY = 'comment_text'

preprocessor = TextPreProcessor(embedding_type='GLOVE')


def pre_process(df, mode='train'):
    print('Running pre-processing...')
    new_df = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    rows = []

    # This is not my best work but it was the only way i could get it all working.
    # @TODO Revist with a fresh perspective.
    for index, tweet in tqdm(df.iterrows()):
        try:
            tweet[TEXT_KEY] = preprocessor.preprocess(tweet[TEXT_KEY], mode)
            rows.append(tweet)
        except:
            pass

    new_df = new_df.append(rows)

    return new_df


def load_data(path, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN, vocab=None):
    df = pd.read_csv(path)

    preprocessor.load_vocab(vocab)

    data_set = pre_process(df)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(data_set[TEXT_KEY])

    x_train = pad_sequences(tokenizer.texts_to_sequences(data_set[TEXT_KEY].fillna(' ').values), maxlen=sequence_length)
    y_train = data_set[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    print('Number of Data Samples:' + str(len(x_train)))
    print('Number of Classes: ' + str(6))

    return (x_train, y_train), tokenizer


def load_data_split(path, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN, vocab=None):
    (x_train, y_train), tokenizer = load_data(path, max_features, sequence_length, vocab=vocab)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, random_state=RANDOM_SEED)

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), tokenizer


def load_data_folds(path, folds=10, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN, vocab=None):
    (x_train, y_train), tokenizer = load_data(path, max_features, sequence_length, vocab=vocab)

    kfold = KFold(n_splits=folds, random_state=RANDOM_SEED)
    folds = kfold.split(x_train)

    return (x_train, y_train), folds, tokenizer


def load_test_data(path, tokenizer, sequence_length=MAX_SEQ_LEN):
    test_set = pd.read_csv(path)
    test_set = pre_process(test_set, mode='test')
    test_set = test_set[TEXT_KEY].fillna(' ').values

    test_set = tokenizer.texts_to_sequences(test_set)
    test_set = pad_sequences(test_set, maxlen=sequence_length)

    return test_set


def load_sample_submission(path):
    submission = pd.read_csv(path)
    return submission
