
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()

nb_classifier.fit(training_set, labels)

pred_labels = nb_classifier.predict(test_set)

class naiveBayes(object):
    """
    This class performs Naive Bayes classification for word-count document features.
    """
    def __init__(self):
        """
        Initialize a Naive Bayes classifier.
        """
        pass

    def fit(self,X,Y):
        """
        Fit the parameters according to the labeled training data (X,Y).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Each row is the word-count vector for one of the documents
        Y : ndarray of shape (n_samples,)
            Gives the class label for each instance of training data. Assume class labels are in {0,1,...,k-1} where k is the number of classes.
        """
        # get prior class probabilities P(c_i)
        # (you may wish to store these as a length k vector as a class attribute)

        # get (smoothed) word-class probabilities
        # (you may wish to store these in a (k, n_features) matrix as a class attribute)

        pass

    def predict(self, X):
        """
        Predict the class labels of a set of test data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The test data

        Returns
        -------
        Y : ndarray of shape (n_samples,)
            Gives the classification of each row in X
        """
        pass

>>> # assume train_vectors, train_labels, and test_vectors are defined
>>> from sklearn.naive_bayes import MultinomialNB
>>> mnb = MultinomialNB()
>>> mnb.fit(train_vectors, train_labels)
>>> predicted = mnb.predict(test_vectors)

>>> with open("SpamVocab.txt", 'r') as f:
>>>     vocab = [s.strip() for s in f]

from collections import Counter
def getCountVector(document, vocab):
    """
    Return the count vector for the given document using the given vocabulary.

    Parameters
    ----------
    document : string
		The text of the document, words separated by whitespaces
    vocab : list of length (n) containing strings
		Each vocab word is a string

    Returns
    -------
    counts : ndarray of shape (1,n)
		The word-count vector
    """
    tf = Counter(document.lower().split()) # get frequencies of each word
    counts = np.array([[tf[t] if t in tf else 0 for t in vocab]])
    return counts


#Generate 100 rolls from 1000 dice, some fair and some weighted
#All weighted dice favor 3 and 4

import numpy as np
import random

prob_fair = 0.7
num_dice = 1000
num_rolls = 100

fair_die = np.array([1, 2, 3, 4, 5, 6])
weighted_die = np.array([1,2,3,3,3,3,4,4,4,4,5,6])

rolls = np.zeros((num_dice, num_rolls))
label = np.zeros((num_dice,1))

for i in xrange(num_dice):
    if np.random.random() < prob_fair:
        for j in xrange(num_rolls):
            rolls[i,j] = random.choice(fair_die)
        label[i] = 0
    else:
        for j in xrange(num_rolls):
            rolls[i,j] = random.choice(weighted_die)
        label[i] = 1


from sklearn.naive_bayes import MultinomailNB
mnb_classifier = MultinomialNB
mnb_classifier.fit(rolls,label)

roll = np.random.randint(1,7,size=100)
mnb.predict(roll)
