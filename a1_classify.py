from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os
import csv
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import random
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from scipy import stats

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # print ('TODO')
    accuracy = np.trace(C) / np.sum(C)
    return accuracy

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    # print ('TODO')
    recall = []
    for i in range(C.shape[0]):
      recall.append(C[i,i] / np.sum(C[i,:]))
    return recall

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    # print ('TODO')
    precision = []
    for i in range(C.shape[1]):
      precision.append(C[i,i] / np.sum(C[:,i]))
    return precision
    
def get_classifier(num):
    # 1. SVC: support vector machine with a linear kernel.
    if (num == 1):
        return SVC(kernel='linear', max_iter = 1000)
    # 2. SVC: support vector machine with a radial basis function (γ = 2) kernel.
    if (num == 2):
        return SVC(gamma = 2, max_iter = 1000)
    # 3. RandomForestClassifier: with a maximum depth of 5, and 10 estimators.
    if (num == 3):
        return RandomForestClassifier(max_depth = 5, n_estimators = 10)
    # 4. MLPClassifier: A feed-forward neural network, with α = 0.05.
    if (num == 4):
        return MLPClassifier(alpha = 0.05)
    # 5. AdaBoostClassifier: with the default hyper-parameters.
    if (num == 5):
        return AdaBoostClassifier()

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    # print('TODO Section 3.1')
    file = np.load(filename)
    feats = file[file.files[0]]

    X = feats[:, 0:173]
    Y = feats[:, 173]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    output_file = open("a1_3.1.csv", "w", newline="\n")
    csv_writer = csv.writer(output_file)
    iBest = -1
    best_accuracy = 0
    for i in range(1, 6):
        classifier = get_classifier(i)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        C = confusion_matrix(y_test, y_pred)
        acc = accuracy(C)
        rec = recall(C)
        pre = precision(C)
        row = [i]
        row.append(acc)
        row += rec
        row += pre
        row = np.append(row, C)
        csv_writer.writerow(row)

        if (acc > best_accuracy):
            iBest = i
            best_accuracy = acc
    output_file.close()       
    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    # print('TODO Section 3.2')
    classifier = get_classifier(iBest)
    train_size = [1000, 5000, 10000, 15000, 20000]
    X_1k = []
    y_1k = []
    data = []

    for size in train_size:
        idx = random.randint(0, y_train.size - size)
        curr_X_train = X_train[idx:idx+size, ]
        curr_y_train = y_train[idx:idx+size, ]

        X_1k = curr_X_train[:1000, ]
        y_1k = curr_y_train[:1000, ]

        classifier.fit(curr_X_train, curr_y_train)
        y_pred = classifier.predict(X_test)
        C = confusion_matrix(y_test, y_pred)
        data.append(accuracy(C))


    output_file = open("a1_3.2.csv", "w", newline="\n")
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(data)
    output_file.close()
    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    # print('TODO Section 3.3')
    num_of_features = [5, 10, 20, 30, 40, 50]

    output_file = open("a1_3.3.csv", "w", newline="\n")
    csv_writer = csv.writer(output_file)
    X_32k = []
    X_32k_test = []
    top_features = []
    for k in num_of_features:
        selector = SelectKBest(f_classif, k)
        X_new = selector.fit_transform(X_train, y_train)
        pp = selector.pvalues_
        int_index = selector.get_support(indices=True)
        print(str(k) + " feature index for 32K: ", int_index)
        if (k == 5):
            top_features = int_index
        associated_pp = [pp[int_index[i]] for i in range(len(int_index))]
        row = [k]
        row = np.append(row, associated_pp)
        csv_writer.writerow(row)
        # get training and test data for 5-fold
        if (k == 5):
            X_32k = X_new
            X_32k_test = np.zeros((X_test.shape[0], len(int_index)))
            for i in range(0, len(int_index)):
                X_32k_test[:, i] = X_test[:, int_index[i]]


    data = []
    # 32ks
    classifier = get_classifier(i)
    classifier.fit(X_32k, y_train)
    y_pred = classifier.predict(X_32k_test)
    C = confusion_matrix(y_test, y_pred)
    data.append(accuracy(C))

    # 1k 
    selector_1k = SelectKBest(f_classif, k=5)
    X_1k_new = selector_1k.fit_transform(X_1k, y_1k)
    int_index_1k = selector_1k.get_support(indices=True)
    print("5 feature index for 1K: ", int_index_1k)
    pp_1k = selector_1k.pvalues_
    associated_pp_1k = [pp_1k[int_index_1k[i]] for i in range(len(int_index_1k))]
    print("p values for 1k: ", associated_pp_1k)
    X_1k_test = np.zeros((X_test.shape[0], len(int_index_1k)))
    for i in range(0, len(int_index_1k)): 
        X_1k_test[:, i] = X_test[:, int_index_1k[i]]
    classifier.fit(X_1k_new, y_1k)
    y_pred = classifier.predict(X_1k_test)
    C = confusion_matrix(y_test, y_pred)
    data.append(accuracy(C))

    csv_writer.writerow(data)

    # find best 5 features for 32K
    feats_names = open('/u/cs401/A1/feats/feats.txt', 'r').read().strip().split("\n")
    row = []
    for i in range(len(top_features)):
        idx = top_features[i]
        if (idx < 29):
            row.append(idx)
        else:
            row.append(feats_names[idx-29])
    csv_writer.writerow(row)
    output_file.close()
    return

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    # print('TODO Section 3.4')
    file = np.load(filename)
    feats = file[file.files[0]]
    X = feats[:, 0:173]
    Y = feats[:, 173]
    k_fold = KFold(n_splits = 5, shuffle = True)

    output_file = open("a1_3.4.csv", "w", newline="\n")
    csv_writer = csv.writer(output_file)
    
    accuracies = []
    for c in range(1, 6):
        classifier = get_classifier(c)
        data = []
        for train_index ,test_index in k_fold.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            conf = confusion_matrix(y_test, y_pred)
            data.append(accuracy(conf))
        csv_writer.writerow(data)
        accuracies.append(data)

    accuracy_best_i = np.array(accuracies[i - 1], dtype=float)
    row = []
    for a in range(len(accuracies)):
        if (a != i - 1):
            acc = np.array(accuracies[a], dtype=float)
            S = stats.ttest_rel(accuracy_best_i, acc)
            row.append(S)
    csv_writer.writerow(row)
    output_file.close() 
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    print(args)
    X_train, X_test, y_train, y_test,iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)
