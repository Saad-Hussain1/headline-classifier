
import random
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus


def load_data():
    '''
    Loads real and fake headlines and splits data into 70% training, 15% validation, 15% test
    Assumes 'clean_fake.txt' and 'clean_real.txt' are in ./data/
    Output: (X_train, X_test, X_valid, y_train, y_test, y_valid, data_vec, labels)
    '''
    with open('./data/clean_fake.txt') as f:
        fake = f.read()
    with open('./data/clean_real.txt') as f:
        real = f.read()
    fake = fake.split('\n')
    real = real.split('\n')
    data = real + fake

    # Create label vector for data; 0 = fake, 1 = real
    labels = np.zeros(len(data))
    for i in range(len(real)):
        labels[i] = 1

    # Vectorize data for splitting
    vectorizer = CountVectorizer()
    data_vec = vectorizer.fit_transform(data)

    # Split data into 70% training, 15% validation, 15% test
    X_train, X_test, y_train, y_test = train_test_split(data_vec, labels, test_size=0.3)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5)

    return (X_train, X_test, X_valid, y_train, y_test, y_valid, data_vec, labels)


def select_model(X_train, X_test, X_valid, y_train, y_test, y_valid):
    '''
    Finds and returns the best decision tree classifier by tuning hyperparameters
    of split criterion (gini or information gain) and tree max depth
    '''
    accuracies_gini = []
    accuracies_infogain = []
    clf_gini = []
    clf_infogain = []

    max_depths = [10, 20, 30, 40, 50]
    # Create decision trees with max depths 10,20,30,40,50
    for i in max_depths:
        # Create trees with gini criterion
        clf_g = DecisionTreeClassifier(criterion='gini', max_depth=i)
        clf_g.fit(X_train, y_train)
        y_pred = clf_g.predict(X_valid)
        # Calculate error
        err = 0
        for j in range(len(y_pred)):
            if y_pred[j] != y_valid[j]:
                err  += 1
        accuracies_gini.append( 100*(len(y_pred)-err)/len(y_pred) )
        clf_gini.append(clf_g)

        # Create trees with infogain criterion
        clf_ig = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        clf_ig.fit(X_train, y_train)
        y_pred = clf_ig.predict(X_valid)
        # Calculate error
        err = 0
        for j in range(len(y_pred)):
            if y_pred[j] != y_valid[j]:
                err += 1
        accuracies_infogain.append( 100*(len(y_pred)-err)/len(y_pred) )
        clf_infogain.append(clf_ig)

    clf_trees = clf_gini + clf_infogain
    accuracies = accuracies_gini + accuracies_infogain
    best_clf = clf_trees[accuracies.index(max(accuracies))]

    # Print accuracies for the 10 trees tested for comparison
    print('Accuracies for gini')
    for i in range(5):
        print('Max depth = {}:\t{}%'.format(max_depths[i], accuracies_gini[i]))
    print('Accuracies for infogain')
    for i in range(5):
        print('Max depth = {}:\t{}%'.format(max_depths[i], accuracies_infogain[i]))

    return best_clf


def compute_information_gain(labels):
    #H(Y) -- (entropy for info gain calc):
    zeros = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            zeros += 1
    pY1 = zeros/len(labels)
    pY2 = 1-pY1
    H_Y = -pY1*np.log2(pY1) - pY2*np.log2(pY2)


if __name__ == "__main__":
    X_train, X_test, X_valid, y_train, y_test, y_valid, data_vec, labels = load_data()
    best_clf = select_model(X_train, X_test, X_valid, y_train, y_test, y_valid)


