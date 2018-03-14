import re
from wordsegment import load, segment, clean
from nltk import download
from nltk.corpus import stopwords

# pre-processing dict
from src.util import GLOVE_WORD_LOOKUP
from src.regexes import REGEX_LOOKUP

download('stopwords')

REMOVE_STOPWORDS = False
STOPWORDS = set(stopwords.words('english'))

# Preprocessing Regex's
url_regex = re.compile(REGEX_LOOKUP['URL'])
ip_regex = re.compile(REGEX_LOOKUP['IP'])
date_regex = re.compile(REGEX_LOOKUP['DATE'])
emoji_regex = re.compile(REGEX_LOOKUP['EMOJI'])
email_regex = re.compile(REGEX_LOOKUP['EMAIL'])
time_regex = re.compile(REGEX_LOOKUP['TIME'])
money_regex = re.compile(REGEX_LOOKUP['MONEY'])
numbers_regex = re.compile(REGEX_LOOKUP['NUMBERS'])

special_character_regex = re.compile('([\;\:\|•«»\n←→])')

users_regex = re.compile('\[\[.*\]')
# users_regex = re.compile('\[\[User(.*)\|')


s_contraction = re.compile('\'s')
d_contraction = re.compile('\'d')
m_contraction = re.compile('\'m')

ll_contraction = re.compile('\'ll')
nt_contraction = re.compile('n\'t')
ve_contraction = re.compile('\'ve')
re_contraction = re.compile('\'re')

cant_replace = re.compile('(can\'t)')
wont_replace = re.compile('(won\'t)')

apostrophe_normalise_regex = re.compile('[‘´’]')

# Remove all non-contraction apostrophes
apostrophe_removal_regex_glove = re.compile("(?!('s)|('d)|('m)|('ll)|(n't)|('ve)|('re))[']")
apostrophe_removal_regex_fast_text = re.compile("[']")

# alpha_test = re.compile("[^\w\s.,?!%^;:{}=+\-_`~()<>|\[\]]")
alpha_test = re.compile("[^\w\s.,?!<>]")

# arrows = re.compile('[<>_]')

tokenize_punct = re.compile('([.,?!]{1})')
control_chars = re.compile('[\n\t\r\v\f\0]')


repeated_punct = re.compile('([!?.]){2,}')
# elongated_words = re.compile('(.)\1{2,}')
elongated_words = re.compile(r"\b(\S*?)(.)\2{2,}\b")
word_split = re.compile(r'[/\-_\\]')


class TextPreProcessor:
    def __init__(self, embedding_type, vocab=None):
        self.embedding_type = embedding_type
        self.OOVs = {}
        self.corrections = {}
        self.vocab = vocab
        self.word_segmenter = None
        self.spellchecker = None
        # load in the vocab
        self.load_vocab(vocab)

    def load_vocab(self, vocab):
        if vocab is not None:
            self.vocab = vocab
            self.spellchecker = SpellChecker(vocab)
            self.word_segmenter = WordSegmentation()
            print('Vocab Loaded into preprocessor, enabling spell checking and OOV word segmentation...')

            return True
        return False

    def preprocess(self, string, mode='train'):
        string = self.clean(string)
        string = self.perform_spellcheck(string, mode)

        return string

    def clean(self, text):
        text = text.lower()

        # text = arrows.sub(' ', text)
        # Replace newline characters
        text = control_chars.sub(' ', text)

        # Replace ips
        text = ip_regex.sub(' <IP> ', text)
        # Replace URLs
        text = url_regex.sub(' <URL> ', text)
        # Replace Emails
        text = email_regex.sub(' <EMAIL> ', text)
        # Replace User Names
        text = users_regex.sub(' <USER> ', text)
        # Replace Dates/Time
        text = date_regex.sub(' <DATE> ', text)
        text = time_regex.sub(' <TIME> ', text)
        # Replace Numbers
        text = numbers_regex.sub(' <NUMBER> ', text)
        # Replace money symbols
        text = money_regex.sub(' <CURRENCY> ', text)

        text = repeated_punct.sub(' \1 <REPEAT> ', text)

        text = word_split.sub(' ', text)

        text = tokenize_punct.sub(r' \1 ', text)

        # text = elongated_words.sub(r"\1", text)
        text = elongated_words.sub(r"\1\2 <ELONG> ", text)

        # Remove a load of unicode emoji characters
        text = emoji_regex.sub('', text)

        # Replace single quotations with apostrophes.
        text = apostrophe_normalise_regex.sub("'", text)

        # Glove records contractions individually e.g. don't -> do n't, therefore if using glove follow the
        if self.embedding_type == 'GLOVE' or True:
            text = cant_replace.sub(' can not ', text)
            text = wont_replace.sub(' will not ', text)

            # Replace contractions
            text = s_contraction.sub(" is ", text)
            text = d_contraction.sub(" would ", text)
            text = m_contraction.sub(" am ", text)

            text = ll_contraction.sub(" will ", text)
            text = nt_contraction.sub(" not ", text)
            text = ve_contraction.sub(" have ", text)
            text = re_contraction.sub(" are ", text)

            # Remove all now non-essential apostrophes.
            # text = apostrophe_removal_regex_glove.sub('', text)
        else:
            # Other embeddings e.g. fast text just remove punctuation.
            text = apostrophe_removal_regex_fast_text.sub('', text)

        # Remove some special characters
        text = special_character_regex.sub(' ', text)
        # Replace numbers and symbols with language
        text = text.replace('&', ' and ')
        # Remove punctuation.
        text = alpha_test.sub(' ', text)
        # Remove multi spaces
        text = re.sub('\s+', ' ', text)
        # Remove ending space if any
        if len(text) > 1:
            text = re.sub('\s+$', '', text)

        return text

    def perform_spellcheck(self, string, mode):
        # Split the string to replace apostrophes etc.
        s = string.split()
        new_string = []

        for word in s:
            if REMOVE_STOPWORDS and word in STOPWORDS:
                continue

            if word in GLOVE_WORD_LOOKUP:
                new_string.append(GLOVE_WORD_LOOKUP[word])
            elif self.vocab is not None and word not in self.vocab:
                # We assume that if we have a vocab and the word is not in it, it is either misspelled or not segmented.
                if word in self.corrections:
                    new_string.append(self.corrections[word])
                    continue

                if mode == 'test':
                    continue

                # If all the words are in the vocab, we assume that we have segmented correctly
                # and therefore we add it to the list
                segmented = self.word_segmenter.segment(word)
                all_in_vocab = True
                for seg in segmented:
                    if seg not in self.vocab:
                        all_in_vocab = False
                        break

                if all_in_vocab:
                    self.corrections[word] = ' '.join(segmented)
                    new_string.extend(segmented)
                else:
                    new_string.append(word)
                #
                # spell_checked = self.spellchecker.correction(word)
                # if spell_checked in self.vocab:
                #     new_string.append(spell_checked)
                #     self.corrections[word] = spell_checked
                #     continue
            else:
                new_string.append(word)

            # if self.corpus_stats[word] > self.word_count_min or word in self.embedding_index:
            #     new_string.append(word)

        # Remove multiple spaces
        s = ' '.join(new_string)
        return s


# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
class SpellChecker:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_rank = self.generate_ranks()

    def generate_ranks(self):
        word_ranks = {}
        for i, word in enumerate(self.vocab):
            word_ranks[word] = i
        return word_ranks

    def P(self, word):
        "Probability of `word`."
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.word_rank.get(word, 0)

    def correction(self, word):
        "Most probable spelling correction for word."
        if len(word) < 30:
            return max(self.candidates(word), key=self.P)
        return word

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.word_rank)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


# Acts as a wrapper around https://github.com/grantjenks/python-wordsegment
class WordSegmentation:
    def __init__(self):
        load()

    def segment(self, word):
        cleaned = clean(word)
        segmented = segment(cleaned)
        return segmented
