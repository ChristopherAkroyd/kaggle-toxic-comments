import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# Tokenize and sequence padding.
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

RANDOM_SEED = 59185
MAX_SEQ_LEN = 250
TEXT_KEY = 'comment_text'

# Smile, Laugh, Love, Wink emoticon regex : :), : ), :-), (:, ( :, (-:, :'), :D, : D, :-D, xD, x-D, XD, X-D,
# <3, ;-), ;), ;-D, ;D, (;, (-;
emo_pos_regex = re.compile('(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))|(:\s?D|:-D|x-?D|X-?D)|(<3)|(;-?\)|;-?D|\(-?;)')
# Sad & Cry emoticon regex: :-(, : (, :(, ):, )-:, :,(, :'(, :"(
emo_neg_regex = re.compile('(:\s?\(|:-\(|\)\s?:|\)-:)|(:,\(|:\'\(|:"\()')
# Turns Yaaaayyy into yay.
collapse_letters_regex = re.compile('(.)\1+')


def pre_process(df):
    new_df = pd.DataFrame.from_items([(name, pd.Series(data=None, dtype=series.dtype)) for name, series in df.iteritems()])
    rows = []

    # This is not my best work but it was the only way i could get it all working.
    # @TODO Revist with a fresh perspective.
    for index, tweet in df.iterrows():
        try:
            tweet[TEXT_KEY] = emo_pos_regex.sub('<EMO_POS>', tweet[TEXT_KEY])
            tweet[TEXT_KEY] = emo_neg_regex.sub('<EMO_NEG>', tweet[TEXT_KEY])
            tweet[TEXT_KEY] = collapse_letters_regex.sub(r'\1\1', tweet[TEXT_KEY])
            rows.append(tweet)
        except:
            pass

    new_df = new_df.append(rows)

    return new_df


def load_data(path, max_features=5000):
    df = pd.read_csv(path)

    print(len(df))

    data_set = pre_process(df)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(data_set[TEXT_KEY])
    word_index = tokenizer.word_index

    X = pad_sequences(tokenizer.texts_to_sequences(data_set[TEXT_KEY].fillna("fillna").values), maxlen=MAX_SEQ_LEN)
    y = data_set[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values


    print('Number of Data Samples:' + str(len(X)))

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    num_classes = 6

    return (np.array(x_train), np.array(y_train)), (np.array(x_val), np.array(y_val)), word_index, num_classes, tokenizer


def load_test_data(path, tokenizer):
    test_set = pd.read_csv(path)
    test_set = test_set[TEXT_KEY].fillna("fillna").values
    test_set = tokenizer.texts_to_sequences(test_set)
    test_set = pad_sequences(test_set, maxlen=MAX_SEQ_LEN)

    return test_set


def load_sample_submission(path):
    submission = pd.read_csv(path)
    return submission
