import numpy as np
from tqdm import tqdm


def read_embeddings_file(path, embedding_type):
    embedding_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # First line is num words/vector size.
            if i == 0 and embedding_type == 'FAST_TEXT':
                continue
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    return embedding_index


def load_embedding_matrix(embeddings_index, word_index, embedding_dimensions):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimensions))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_embeddings(path, embedding_type, word_index, embedding_dimensions=300):
    if embedding_type == 'GLOVE' or embedding_type == 'FAST_TEXT':
        print('Loading {} embeddings from file...'.format(embedding_type))
    else:
        print('Generating random uniform embeddings...')
        return np.random.uniform(low=-0.05, high=0.05, size=(len(word_index) + 1, embedding_dimensions))

    embedding_index = read_embeddings_file(path, embedding_type)

    embedding_matrix = load_embedding_matrix(embedding_index, word_index, embedding_dimensions)

    print('Vocabulary Size: %s words.' % len(embedding_matrix))

    del embedding_index

    return embedding_matrix

