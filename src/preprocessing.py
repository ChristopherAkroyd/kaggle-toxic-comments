import re
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
    def __init__(self, embedding_type):
        self.embedding_type = embedding_type

    def preprocess(self, string):
        string = self.clean(string)
        string = self.perform_spellcheck(string)

        return string

    def clean(self, s):
        s = s.lower()

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

    def perform_spellcheck(self, string):
        # Split the string to replace apostrophes etc.
        s = string.split()
        new_string = []

        for word in s:
            if word in GLOVE_WORD_LOOKUP:
                new_string.append(GLOVE_WORD_LOOKUP[word])
            else:
                new_string.append(word)

            # if self.corpus_stats[word] > self.word_count_min or word in self.embedding_index:
            #     new_string.append(word)

        # Remove multiple spaces
        s = ' '.join(new_string)
        return s
