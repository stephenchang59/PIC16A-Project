# Created_Class
import sklearn
from sklearn import tree, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
class classifier:
    def __init__(self, model_name, train, test, X_vars, y, hyperparameter_decision_tree = 5, hyperparameter_knn = 3):
        '''
        Creates either a KNN or Decision tree model based on users picked columns of interest and hyperparameters 
   
        Args: 
          model_name -- string "Decision Tree" or "KNN"
          train -- training data 
          test -- testing data 
          X_vars -- list of columns/predictors
          y -- string varaible representing column to be predicted
          hyperparameter_decision_tree -- the max_depth hyperparameter for decicision trees (default 5)
          hyperparameter_knn -- the num_neighbors hyperparameter for knn (default 3)
        
        Returns: none
        '''
        np.random.seed(1)
        self.model_name = model_name
        self.train = train
        self.test = test
        self.X_vars = X_vars
        self.y = y
        self.hyperparameter_decision_tree =  hyperparameter_decision_tree
        self.hyperparameter_knn = hyperparameter_knn
        if self.model_name == "Decision Tree":
            self.clf = tree.DecisionTreeClassifier(max_depth = self.hyperparameter_decision_tree) #Constructs decision tree model
        elif self.model_name == "Knn": 
            if any((item == "Sex") or (item == "Island") for item in  self.X_vars): #Checks if user provided categorical variables for knn
                raise ValueError("You cant use categorical vars")          
            #Standardizes training and test data for the Knn model 
            self.train[self.X_vars] = (self.train[self.X_vars] - self.train[self.X_vars].mean())/self.train[self.X_vars].std()  
            self.test[self.X_vars] = (self.test[self.X_vars] - self.test[self.X_vars].mean())/self.test[self.X_vars].std()
            self.clf = KNeighborsClassifier(n_neighbors = self.hyperparameter_knn) #Constructs knn model 
        else: 
            raise TypeError("The model must be Decision Tree or Knn")
            
            
           
    def training_accuracy(self):
        '''
        Fits the training data and calculates the training accuracy
        
        Args: none
        
        Returns: The training accuracy
        '''
        np.random.seed(1)
        self.clf.fit(self.train[self.X_vars], self.train[self.y]) #Fits the model to the training data 
        return(self.clf.score(self.train[self.X_vars], self.train[self.y])) #Returns the training accuracy  
    
    
    def cross_validate(self):
        '''
        Calculates the cross validation accuracy
        
        Args: none
        
        Returns: Cross validation accuracy
        '''
        np.random.seed(1)
        return(cross_val_score(self.clf, self.train[self.X_vars], self.train[self.y], cv = 5).mean()) #Returns the cross validation accuracy
    
    
    def cross_validate_for_hyperparameter(self, lower_bound, upper_bound):
        '''
        Finds the best hyperparameter among a user specified set of hyperparameters by providing a graphic of training and cross valdiations
        Prints the best hyperparameter and a graphic 
       
        Args: 
          lower_bound -- an int representing the lower bound of the hyperparamter to try (typically 1)
          upper_bound -- an int representing the upper bound of the hyperparamter to try
        
        Returns: none
        ''' 
        np.random.seed(1)
        fig, ax = plt.subplots(1, figsize = (10,8))
        crossValList = [] #creates empty list for cross validation scores
        trainList = [] # empty list for training scores
        list_x = list(range(lower_bound, upper_bound + 1)) # list for hyperparameter values
        for i in range(lower_bound, upper_bound + 1):
            if self.model_name == "Decision Tree":
                Z = classifier("Decision Tree", self.train, self.test, self.X_vars, self.y, hyperparameter_decision_tree = i) 
            if self.model_name == "Knn":
                Z = classifier("Knn", self.train, self.test, self.X_vars, self.y, hyperparameter_knn = i)
            trainList.append(Z.training_accuracy()) # appends current models training scores to trainList
            crossValList.append(Z.cross_validate()) # appends current models cross validation score to crossValList
             
        print("Hyperparameter that maximizes cross valdiation ", crossValList.index(max(crossValList)) + lower_bound, " with accuracy score of  ", crossValList[crossValList.index(max(crossValList))]) #Prints out the best hyperparameter and its cross valdiation score 
        ax.scatter(list_x, trainList, label = 'Training score') # Creates scatterplot graphic
        ax.scatter(list_x, crossValList, label = 'Cross Validation score')
        ax.set(xlabel = 'Hyperparameter', ylabel = 'Cross Validation Score')
        plt.legend(fontsize = 20)
        plt.title('Cross Validation for Hyperparameter')
        plt.show() 
        
        
        
    def testing_accuracy(self):
        '''
        Prints the testing accuracy and confusion matrix heatmap
        
        Args: none
        
        Returns: none 
        '''
        print(self.clf.score(self.test[self.X_vars], self.test[self.y])) #Prints testing accuracy
        actual=(self.test[self.y]).to_numpy()
        pred= self.clf.predict(self.test[self.X_vars])
        matrix= (confusion_matrix(actual, pred, labels=[0,1,2])) #Constructs confusion matrix with actual and predicted values 
        fig, ax = plt.subplots(1, figsize = (10,8))
        sns.heatmap(matrix, annot= True, cmap= "Blues", fmt= ".0f"); #Constructs heat map
        ax.set_xticklabels(["Adelie", "Gentoo", "Chinstrap"])
        ax.set_yticklabels(["Adelie", "Gentoo", "Chinstrap"])
        ax.set_ylabel("predicted value")
        ax.set_xlabel("true value")