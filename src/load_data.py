import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold

# Tokenize and sequence padding.
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# preprocessing dict
from src.util import get_word_lookup_dict


RANDOM_SEED = 233
MAX_SEQ_LEN = 100
MAX_FEATS = 15000
REMOVE_NUMBERS = False
TEXT_KEY = 'comment_text'
WORD_LOOKUP = get_word_lookup_dict()

url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
ip_regex = re.compile('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
isolate_punctuation_regex = re.compile('([\'\"\.\(\)\!\?\-\\\/\,])')
special_character_regex = re.compile('([\;\:\|•«\n])')
punctuation_regex = re.compile('[^\w\s]')
numbers_regex = re.compile('^\d+\s|\s\d+\s|\s\d+$')
users_regex = re.compile('\[\[.*\]')


def normalize(s):
    # Replace ips
    s = ip_regex.sub(' <IP> ', s)
    # Replace URLs
    s = url_regex.sub(' <URL> ', s)
    # Replace User Names
    s = users_regex.sub(' <USER> ', s)
    # Remove numbers - Idea is they have little effect on toxicity and are basically valueless.
    if REMOVE_NUMBERS:
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
    # Split the string to replace apostrophes etc.
    s = s.split()
    s = [WORD_LOOKUP[word] if word in WORD_LOOKUP else word for word in s]

    # Remove multiple spaces
    s = ' '.join(s)

    return s


def pre_process(df):
    print('Running pre-processing...')
    new_df = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    rows = []

    # This is not my best work but it was the only way i could get it all working.
    # @TODO Revist with a fresh perspective.
    for index, tweet in tqdm(df.iterrows()):
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

    x_train = pad_sequences(tokenizer.texts_to_sequences(data_set[TEXT_KEY].fillna(' ').values), maxlen=sequence_length)
    y_train = data_set[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    return (x_train, y_train), tokenizer


def load_data_split(path, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN):
    (x_train, y_train), tokenizer = load_data(path, max_features, sequence_length)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05, random_state=RANDOM_SEED)

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), tokenizer


def load_data_folds(path, folds=10, max_features=MAX_FEATS, sequence_length=MAX_SEQ_LEN):
    (x_train, y_train), tokenizer = load_data(path, max_features, sequence_length)

    kfold = KFold(n_splits=folds, random_state=RANDOM_SEED)
    folds = kfold.split(x_train)

    return (x_train, y_train), folds, tokenizer


def load_test_data(path, tokenizer, sequence_length=MAX_SEQ_LEN):
    test_set = pd.read_csv(path)
    test_set = pre_process(test_set)
    test_set = test_set[TEXT_KEY].fillna(' ').values

    test_set = tokenizer.texts_to_sequences(test_set)
    test_set = pad_sequences(test_set, maxlen=sequence_length)

    return test_set


def load_sample_submission(path):
    submission = pd.read_csv(path)
    return submission
