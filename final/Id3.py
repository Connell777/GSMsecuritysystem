#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import defaultdict, Counter
from math import log as logarithm
from Features import feat

class ID3:
    def __init__(self, max_depth=float("inf"), min_gain=0, depth=0):
        """
        Arguments:
            max_depth: After eaching this depth, the current node is turned into a leaf which predicts
                the most common label. This limits the capacity of the classifier and helps combat overfitting
            min_gain: The minimum gain a split has to yield. Again, this helps overfitting
            depth: Let's the current node know how deep it is into the tree, users usually don't need to set this
        """
        
        self.depth = depth
        self.max_depth = max_depth
        self.min_gain = min_gain
        
        # ID3 nodes are either nodes that make a decision or leafs which constantly predict the same result
        # We represent both possibilities using `ID3` objects and set `self.leaf` respectively
        self.leaf = False
        self.value = None
        
        self.children = {}
        self.feature = 0
    
    def fit(self, X, y):
        """
        Creates a tree structure based on the passed data
        
        Arguments:
            X: numpy array that contains the features in its rows
            y: numpy array that contains the respective labels
        """
        
        self.counts = Counter(y)
        self.most_common_label = self.counts.most_common()[0][0]
        
        # If there is only one class left, turn this node into a leaf
        # and always return this one value
        if len(set(y)) == 1:
            self.leaf = True
            self.value = y[0]
        # If the tree is getting to deep, turn this node into a leaf
        # and always predict the most common value
        elif self.depth >= self.max_depth:
            self.leaf = True
            self.value = self.most_common_label
        # Otherwise, look for the most informative feature and do a split on its possible values
        else:
            self.feature = self._choose_feature(X, y)
            
            # If no feature is informative enough, turn this node into a leaf
            # and always predict the most common value
            if self.feature is None:
                self.leaf = True
                self.value = self.most_common_label
            else:
                for value, (Xi, yi) in self._partition(X, y, self.feature).items():
                    child = ID3(max_depth=self.max_depth, depth=self.depth+1)
                    child.fit(Xi, yi)
                    self.children[value] = child
    
    def predict_single(self, x):
        """
        Predict the class of a single data point x by either using the value encoded in a leaf
        or by following the tree structure recursively until a leaf is reached
        
        Arguments:
            x: individual data point
        """
        
        if self.leaf:
            return self.value
        else:
            value = x[self.feature]
            
            if value in self.children:
                return self.children[value].predict_single(x)
            else:
                return self.most_common_label
        
    def predict(self, X):
        """
        Predict the results for an entire dataset
        
        Arguments:
            X: numpy array that contains each data point in a row
        """
        
        return [self.predict_single(x) for x in X]
    
    def score(self, X, y):
        """
        Returns the accuracy for predicting the given dataset X
        """
        
        correct = sum(self.predict(X) == y)
        return float(correct) / len(y)
        
    def _choose_feature(self, X, y):
        """
        Finds the most informative feature to split on and returns its index.
        If no feature is informative enough, `None` is returned
        """
        
        best_feature = 0
        best_feature_gain = -float("inf")
        
        for i in range(X.shape[1]):
            gain = self._information_gain(X, y, i)
                        
            if gain > best_feature_gain:
                best_feature = i
                best_feature_gain = gain
        
        if best_feature_gain > self.min_gain:
            return best_feature
        else:
            return None
    
    def _information_gain(self, X, y, feature):
        """
        Calculates the information gain achieved by splitting on the given feature
        """
        
        result = self._entropy(y)
        
        summed = 0
        
        for value, (Xi, yi) in self._partition(X, y, feature).items():
            summed += float(len(yi)) / len(y) * self._entropy(yi)
        
        result -= summed
        
        return result
    
    def _entropy(self, X):
        """
        Calculates the Shannon entropy on the given data X
        
        Arguments:
            X: An iterable for feature values. Usually, this is now a 1D list
        """
        
        summed = 0
        counter = Counter(X)

        for value in counter:
            count = counter[value]
            px = count / float(len(X))
            summed += px * logarithm(1. / px, 2)
        
        return summed        
    
    def _partition(self, X, y, feature):
        """
        Partitioning is a common operation needed for decision trees (or search trees).
        Here, a partitioning is represented by a dictionary. The keys are values that the feature
        can take. Under each key, we save a tuple (Xi, yi) that represents all data points (and their labels)
        that have the respective value in the specified feature.
        """
        
        partition = defaultdict(lambda: ([], []))
        
        for Xi, yi in zip(X, y):
            bucket = Xi[feature]
            partition[bucket][0].append(Xi)
            partition[bucket][1].append(yi)
        
        partition = dict(partition)
            
        for feature, (Xi, yi) in partition.items():
            partition[feature] = (np.array(Xi), np.array(yi))
            
        return partition


# In[11]:


import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
def preprocess(data, encode_labels=False):
    X = data.drop(["Result"], 1)    
    
    if encode_labels: # for sklearn
        X = X.apply(LabelEncoder().fit_transform)
        
    X.head()    
    return X.to_numpy()


# In[12]:

def train():
    data =pd.read_csv("/home/pi/Desktop/tudfyp/uncalread data/data/tapdatacolt.csv")
    y = data["Result"].to_numpy()
    X = preprocess(data)


    # In[40]:


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = ID3()
    clf.fit(X_train, y_train)


    # In[41]:


    print ("train accuracy = %.5f" % clf.score(X_train, y_train))
    print ("test accuracy = %.5f" % clf.score(X_test, y_test))


    # In[42]:


    for max_depth in range(15):
        clf = ID3(max_depth=max_depth)
        clf.fit(X_train, y_train)
        print( max_depth, clf.score(X_test, y_test))


def pred():
    clf=ID3()
    data = []
    X_new = feat(data)
    if(clf.predict(X_new)==[1]):
        return True
        

train()


