import pandas as pd
import numpy as np
import re
import tqdm
from sklearn.model_selection import train_test_split, KFold

# Tokenize and sequence padding.
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

RANDOM_SEED = 233
MAX_SEQ_LEN = 100
MAX_FEATS = 15000
TEXT_KEY = 'comment_text'

url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
ip_regex = re.compile('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
isolate_punctuation_regex = re.compile('([\'\"\.\(\)\!\?\-\\\/\,])')
special_character_regex = re.compile('([\;\:\|•«\n])')
punctuation_regex = re.compile('[^\w\s]')
numbers_regex = re.compile('^\d+\s|\s\d+\s|\s\d+$')


def normalize(s):
    # Replace ips
    s = ip_regex.sub(' <IP> ', s)
    # Replace URLs
    s = url_regex.sub(' <URL> ', s)
    # Remove numbers - Idea is they have little effect on toxicity and are basically valueless.
    s = numbers_regex.sub(' ', s)
    # Isolate punctuation
    s = isolate_punctuation_regex.sub(r' \1 ', s)
    # Remove some special characters
    s = special_character_regex.sub(' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    # Remove punctuation.
    s = punctuation_regex.sub(' ', s)
    # Replace newline characters
    s = s.replace('\n', ' ')
    s = s.replace('\n\n', ' ')
    # Remove multiple spaces
    s = ' '.join(s.split())
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

    x_train = pad_sequences(tokenizer.texts_to_sequences(data_set[TEXT_KEY].fillna(' ').values), maxlen=sequence_length)
    y_train = data_set[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    return (x_train, y_train), word_index, tokenizer


def load_data_split(path, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN):
    (x_train, y_train), word_index, tokenizer = load_data(path, max_features, sequence_length)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, random_state=RANDOM_SEED)
    num_classes = 6

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), word_index, num_classes, tokenizer


def load_data_folds(path, folds=10, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN):
    (x_train, y_train), word_index, tokenizer = load_data(path, max_features, sequence_length)

    kfold = KFold(n_splits=folds, random_state=RANDOM_SEED)
    folds = kfold.split(x_train)
    num_classes = 6

    return (x_train, y_train), folds, word_index, num_classes, tokenizer


def load_test_data(path, tokenizer, sequence_length=MAX_SEQ_LEN):
    test_set = pd.read_csv(path)
    new_df = pd.DataFrame.from_items(
        [(name, pd.Series(data=None, dtype=series.dtype)) for name, series in test_set.iteritems()])
    rows = []

    for index, tweet in test_set.iterrows():
        try:
            tweet[TEXT_KEY] = normalize(tweet[TEXT_KEY])
            rows.append(tweet)
        except:
            pass

    new_df = new_df.append(rows)
    test_set = new_df
    test_set = test_set[TEXT_KEY].fillna(' ').values

    test_set = tokenizer.texts_to_sequences(test_set)
    test_set = pad_sequences(test_set, maxlen=sequence_length)

    return test_set


def load_sample_submission(path):
    submission = pd.read_csv(path)
    return submission
