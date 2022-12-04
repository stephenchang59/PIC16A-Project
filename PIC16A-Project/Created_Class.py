import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree, preprocessing
import numpy as np
import sklearn
from sklearn.model_selection import cross_val_score, train_test_split
import os
import importlib

# Created_Class
class classifier:
    '''
    Creates and fits either KNN or Decision tree model based on users picked columns of interest and hyperparameters 
    Default hyperparameters: Decision Tree -- max depth: 5, KNN -- neighbors: 3
    
    Args: model_name -- string "Decision Tree" or "KNN"
          X_vars -- list of columns/predictors
          y -- string varaible representing column to be predicted
          hyperparameter_decision_tree -- the max_depth hyperparameter for decicision trees
          hyperparameter_knn -- the max_depth hyperparameter for knn
    '''
    def __init__(self, model_name, train, test, X_vars, y, hyperparameter_decision_tree = 5, hyperparameter_knn = 3):
        self.model_name = model_name
        self.train = train
        self.test = test
        self.X_vars = X_vars
        self.y = y
        self.hyperparameter_decision_tree =  hyperparameter_decision_tree
        self.hyperparameter_knn = hyperparameter_knn
        if self.model_name == "Decision Tree":
            self.clf = tree.DecisionTreeClassifier(max_depth = self.hyperparameter_decision_tree)
        #IF MODEL NAME == KNN
            #NORMALIZE ALL COLUMNS THAT ARE NUMERIC (DO NOT DO CHANGE COLUMNS LIKE ISLAND, SEX, OR SPECIES)
            #CREATE CLASSIFIER 
        else: 
            raise TypeError
    def training_accuracy(self):
        np.random.seed(1)
        self.clf.fit(self.train[self.X_vars], self.train[self.y])
        return (self.clf.score(self.train[self.X_vars], self.train[self.y]))
        #print(self.clf.score(self.train[self.X_vars], self.train[self.y]))     
    def cross_validate(self):
        np.random.seed(1)
        return (cross_val_score(self.clf, self.train[self.X_vars], self.train[self.y], cv = 5).mean())
        #print(cross_val_score(self.clf, self.train[self.X_vars], self.train[self.y], cv = 5).mean())
        #CREATE GRAPHIC WITH THESE SCORES
    
    def cross_validate_for_hyperparameter(self, lower_bound, upper_bound):
        np.random.seed(1)
        fig, ax = plt.subplots(1, figsize = (10,8))
        crossValList = [] #creates empty list for cross validation scores
        trainList = [] # empty list for training scores
        list_x = list(range(lower_bound, upper_bound + 1)) # list for hyperparameter values
        for i in range(lower_bound, upper_bound + 1):
            if self.model_name == "Decision Tree":
                Z = classifier("Decision Tree", self.train, self.test, self.X_vars, self.y, hyperparameter_decision_tree = i)
            if self.model_name == "Knn":
                Z = classifier("Knn", self.train, self.test, self.X_vars, self.y, hyperparameter_decision_tree = i)
            crossValList.append(Z.cross_validate()) # appends current cross validation scores to crossValList
            trainList.append(Z.training_accuracy()) # appends current cross validation scores to trainList
        ax.scatter(list_x, trainList, label = 'Training score')
        ax.scatter(list_x, crossValList, label = 'Cross Validation score')
        ax.set(xlabel = 'Hyperparameter', ylabel = 'Cross Validation Score')
        plt.legend(fontsize = 20)
        plt.title('Cross Validation for Hyperparameter')
        plt.show()
        
        #DO THIS FOR KNN
    def testing_accuracy(self):
        print(self.clf.score(self.test[self.X_vars], self.test[self.y])) 
        #CONFUSION MATRIX 