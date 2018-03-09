import numpy as np
from tqdm import tqdm


def read_embeddings_file(path, embedding_type):
    embedding_index = {}
    print('Reading {} file...'.format(embedding_type))
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            # First line is num words/vector size.
            if i == 0 and embedding_type == 'FAST_TEXT':
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    return embedding_index


def load_embedding_matrix(embedding_index, word_index, embedding_dimensions):
    print('Generating embedding matrix...')
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimensions))
    oov_count = 0
    for word, i in tqdm(word_index.items()):
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            oov_count += 1

    print('{} Unique words (vocab size).'.format(len(embedding_matrix)))
    print('{} Unique OOV words.'.format(oov_count))

    return embedding_matrix, oov_count


def load_embeddings(path, embedding_type, word_index, embedding_dimensions=300, embedding_index=None):
    if embedding_type == 'GLOVE' or embedding_type == 'FAST_TEXT':
        print('Loading {} embeddings...'.format(embedding_type))
    else:
        print('Generating random uniform embeddings...')
        return np.random.uniform(low=-0.05, high=0.05, size=(len(word_index) + 1, embedding_dimensions))

    if embedding_index is None:
        embedding_index = read_embeddings_file(path, embedding_type)

    embedding_matrix, oov_count = load_embedding_matrix(embedding_index, word_index, embedding_dimensions)

    return embedding_matrix

