import pathlib
from tqdm import tqdm

GLOVE_WORD_LOOKUP = {
    # Domain specific
    "mainpagebg": " ",
    "concernthanks": "concern thanks",
    "gayfrozen": "gay frozen",
    "talkcontribs": "talk contributions",
    "tryin'": "trying",
    "gayfag": "gay fag",
    "f**k": "fuck",
    "s**t": "shit",
    "f***": "fuck",
    "a**": "ass",
    "contribs": "contributions",
    "ai": "am",
    "wo": "will"
}

WORD_LOOKUP = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will",
    "tryin'": "trying",
    "f**k": "fuck",
    "s**t": "shit",
    "f***": "fuck",
    "a**": "ass"
}


def get_save_path(model, directory='./model_checkpoints', fold=None):
    model_name = model.__class__.__name__
    path = directory + '/{}/{}'.format(model_name, model_name)
    # create dirs if they don't exist.
    pathlib.Path(directory + '/{}/'.format(model_name)).mkdir(parents=True, exist_ok=True)

    if fold is not None:
        path = path + '-fold-{}'.format(fold)

    path = path + '.hdf5'

    return path


def get_submission_path(model):
    model_name = model.__class__.__name__
    path = '{}_submission.csv'.format(model_name)
    return path


class CorpusStats:
    def __init__(self, corpus):
        self.corpus_stats = {}
        self.build_corpus_stats(corpus)

    def build_corpus_stats(self, corpus):
        print('Building Corpus Word Stats...')
        for entry in tqdm(corpus):
            cleaned = entry.split(' ')
            for word in cleaned:
                if word in self.corpus_stats:
                    self.corpus_stats[word] += 1
                else:
                    self.corpus_stats[word] = 1
