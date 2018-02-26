import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# Tokenize and sequence padding.
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

RANDOM_SEED = 59185
MAX_SEQ_LEN = 100
MAX_FEATS = 15000
TEXT_KEY = 'comment_text'


def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP> ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s


def pre_process(df):
    new_df = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    rows = []

    # This is not my best work but it was the only way i could get it all working.
    # @TODO Revist with a fresh perspective.
    for index, tweet in df.iterrows():
        try:
            tweet[TEXT_KEY] = normalize(tweet[TEXT_KEY])
            rows.append(tweet)
        except:
            pass

    new_df = new_df.append(rows)

    return new_df


def load_data(path, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN):
    df = pd.read_csv(path)

    data_set = pre_process(df)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(data_set[TEXT_KEY])
    word_index = tokenizer.word_index

    x_train = pad_sequences(tokenizer.texts_to_sequences(data_set[TEXT_KEY].fillna("fillna").values), maxlen=sequence_length)
    y_train = data_set[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    return (x_train, y_train), word_index, tokenizer


def load_data_split(path, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN):
    (x_train, y_train), word_index, tokenizer = load_data(path, max_features, sequence_length)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    num_classes = 6

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), word_index, num_classes, tokenizer


def load_test_data(path, tokenizer, sequence_length=MAX_SEQ_LEN):
    test_set = pd.read_csv(path)
    test_set = test_set[TEXT_KEY].fillna("fillna").values
    test_set = tokenizer.texts_to_sequences(test_set)
    test_set = pad_sequences(test_set, maxlen=sequence_length)

    return test_set


def load_sample_submission(path):
    submission = pd.read_csv(path)
    return submission
