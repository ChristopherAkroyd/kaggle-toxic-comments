import re
from wordsegment import load, segment, clean
# pre-processing dict
from src.util import GLOVE_WORD_LOOKUP
from src.regexes import REGEX_LOOKUP


REMOVE_NUMBERS = False

# Preprocessing Regex's
url_regex = re.compile(REGEX_LOOKUP['URL'])
ip_regex = re.compile(REGEX_LOOKUP['IP'])
date_regex = re.compile(REGEX_LOOKUP['DATE'])
emoji_regex = re.compile(REGEX_LOOKUP['EMOJI'])
email_regex = re.compile(REGEX_LOOKUP['EMAIL'])
time_regex = re.compile(REGEX_LOOKUP['TIME'])
money_regex = re.compile(REGEX_LOOKUP['MONEY'])

special_character_regex = re.compile('([\;\:\|•«»\n←])')
# punctuation_regex = re.compile('[.,\\\/⁄#!°·?%\^"“”˜¯&;:{}=\-–—…‖_`≤~()<>|\[\]]')
users_regex = re.compile('\[\[.*\]')

s_contraction = re.compile('\'s')
d_contraction = re.compile('\'d')
m_contraction = re.compile('\'m')

ll_contraction = re.compile('\'ll')
nt_contraction = re.compile('n\'t')
ve_contraction = re.compile('\'ve')
re_contraction = re.compile('\'re')

apostrophe_normalise_regex = re.compile('[‘´’]')

# Remove all non-contraction apostrophes
apostrophe_removal_regex_glove = re.compile("(?!('s)|('d)|('m)|('ll)|(n't)|('ve)|('re))[']")
apostrophe_removal_regex_fast_text = re.compile("[']")

alphanumeric_only = re.compile("(?!('s)|('d)|('m)|('ll)|(n't)|('ve)|('re))[^\w\s]")

alpha_test = re.compile("[^\w\s.,?!%^;:{}=+\-_`~()<>|\[\]]")


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

    def clean(self, s):
        s = s.lower()

        # s = ' '.join(s.split('-'))
        # s = ' '.join(s.split('_'))

        # Replace ips
        s = ip_regex.sub(' <IP> ', s)
        # Replace URLs
        s = url_regex.sub(' <URL> ', s)
        # Replace Emails
        s = email_regex.sub(' <EMAIL> ', s)
        # Replace User Names
        s = users_regex.sub(' <USER> ', s)
        # Replace Dates/Time
        s = date_regex.sub(' <DATE> ', s)
        s = time_regex.sub(' <TIME> ', s)
        # Replace money symbols
        s = money_regex.sub('', s)

        # Remove a load of unicode emoji characters
        s = emoji_regex.sub('', s)

        # Replace single quotations with apostrophes.
        s = apostrophe_normalise_regex.sub("'", s)

        # Glove records contractions individually e.g. don't -> do n't, therefore if using glove follow the
        if self.embedding_type == 'GLOVE' or True:
            # Replace contractions
            s = s_contraction.sub(" is ", s)
            s = d_contraction.sub(" would ", s)
            s = m_contraction.sub(" am ", s)

            s = ll_contraction.sub(" will ", s)
            s = nt_contraction.sub(" not ", s)
            s = ve_contraction.sub(" have ", s)
            s = re_contraction.sub(" are ", s)

            # Remove all now non-essential apostrophes.
            # s = apostrophe_removal_regex_glove.sub('', s)
        else:
            # Other embeddings e.g. fast text just remove punctuation.
            s = apostrophe_removal_regex_fast_text.sub('', s)

        # Remove some special characters
        s = special_character_regex.sub(' ', s)
        # Numbers are probably value less
        s = re.sub("\d+", " ", s)

        # Replace numbers and symbols with language
        s = s.replace('&', ' and ')
        # Replace newline characters
        s = s.replace('\n', ' ')
        s = s.replace('\n\n', ' ')

        s = s.replace('\t', ' ')
        s = s.replace('\b', ' ')
        s = s.replace('\r', ' ')
        # Remove punctuation.
        s = alpha_test.sub(' ', s)
        # Remove multi spaces
        s = re.sub('\s+', ' ', s)
        # Remove ending space if any
        s = re.sub('\s+$', '', s)

        return s

    def perform_spellcheck(self, string, mode):
        # Split the string to replace apostrophes etc.
        s = string.split()
        new_string = []

        for word in s:
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
        return max(self.candidates(word), key=self.P)

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
