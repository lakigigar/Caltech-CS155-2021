import numpy as np

##########################
# Helper functions/classes
##########################

class WordPair:
    """
    Class representing a pair of words in our vocabulary, along with the cosine similarity
    of the two words.
    """
    def __init__(self, firstWord, secondWord, similarity):
        """
        Initializes the WordPair given two words (strings) and their similarity (float).
        """
        # Ensure that our pair consists of two distinct words
        assert(firstWord != secondWord)
        self.firstWord = firstWord
        self.secondWord = secondWord
        self.similarity = similarity

    def __repr__(self):
        """
        Define the string representation of a WordPair so that a WordPair instance x
        can be displayed using print(x).
        """
        return "Pair(%s, %s), Similarity: %s"%(self.firstWord, self.secondWord, self.similarity)


def sort_by_similarity(word_pairs):
    """
    Given a list of word pair instances, returns a list of the instances sorted
    in decreasing order of similarity.
    """
    return sorted(word_pairs, key=lambda pair: pair.similarity, reverse=True)

def get_similarity(v1, v2):
    """ Returns the cosine of the angle between vectors v1 and v2. """
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    return np.dot(v1_unit, v2_unit)


def load_word_list(path):
    """
    Loads a list of the words from the file at path <path>, removing all
    non-alpha-numeric characters from the file.
    """
    with open(path) as handle:
        # Load a list of whitespace-delimited words from the specified file
        raw_text = handle.read().strip().split()
        # Strip non-alphanumeric characters from each word
        alphanumeric_words = map(lambda word: ''.join(char for char in word if char.isalnum()), raw_text)
        # Filter out words that are now empty (e.g. strings that only contained non-alphanumeric chars)
        alphanumeric_words = filter(lambda word: len(word) > 0, alphanumeric_words)
        # Convert each word to lowercase and return the result
        return list(map(lambda word: word.lower(), alphanumeric_words))

def generate_onehot_dict(word_list):
    """
    Takes a list of the words in a text file, returning a dictionary mapping
    words to their index in a one-hot-encoded representation of the words.
    """
    word_to_index = {}
    i = 0
    for word in word_list:
        if word not in word_to_index:
            word_to_index[word] = i
            i += 1
    return word_to_index

def most_similar_pairs(weight_matrix, word_to_index):
    """
    For each word a in our vocabulary, computes the most similar word b to a, along with the
    cosine similarity of a and b.

    Arguments:
        weight_matrix: The matrix of weights extracted from the hidden layer of a fitted
                       neural network.

        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

    Returns: 
        A list of WordPair instances sorted in decreasing order of similarity,
        one representing each word <vocab_word> and its most similar word.
    """
    word_to_feature_repr = get_word_to_feature_repr(weight_matrix, word_to_index)
    result = []
    for word in word_to_feature_repr:
        result.append(most_similar_word(word_to_feature_repr, word))
    return sort_by_similarity(result)

def most_similar_word(word_to_feature_repr, input_word):
    """
    Given a dictionary mapping words to their feature representations (word_to_feature_repr),
    returns the a WordPair instance corresponding to the word
    whose feature vector is most similar to the feature representation of the
    passed-in word (input_word).
    """
    best_word = None
    best_similarity = 0
    input_vec = word_to_feature_repr[input_word]
    for word, feature_vec in word_to_feature_repr.items():
        similarity = get_similarity(input_vec, feature_vec)
        if similarity > best_similarity and np.linalg.norm(feature_vec - input_vec) != 0:
            best_similarity = similarity
            best_word = word
    return WordPair(input_word, best_word, best_similarity)

def get_word_to_feature_repr(weight_matrix, word_to_index):
    """
    Returns a dictionary mapping each word in our vocabulary to its one-hot-encoded
    feature representation.

    Arguments:
        weight_matrix: The matrix of weights extracted from the hidden layer of a fitted
                       neural network.

        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.
    """
    assert(weight_matrix is not None)
    word_to_feature_repr = {}
    for word, one_hot_idx in word_to_index.items():
        word_to_feature_repr[word] = weight_matrix[one_hot_idx]
    return word_to_feature_repr
