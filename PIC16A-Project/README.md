# PIC16A-Project
Penguins Project
Stephen, Glenda, Armine, and Ryan
For our group project, we decided to do the clustering penguin analysis. We will work with a dataset containing features such as sex, island of origin, species, flipper length, etc. for over 300 penguins. Our goal will be to construct two different models using the KNN algorithm and decision tree/random forest algorithm and compare their performance on predicting the species of penguins. First, we will do EDA to show which features are correlated to our response variable, see the distribution of our data, and check cross-feature correlation. Next, we will use cross-validation to select the best hyperparameters. Finally, we will display our results which will be their accuracy scores on testing data. 

Python Packages Used: matplotlib 3.4.3, sklearn 1.1.3, numpy 1.20.3, pandas 1.3.4, seaborn 0.11.2

Description of Demo File: 
    First, the demo file cleans the data and preforms EDA using the custom function with the user specified columns of interest. 
    Then it constructs decision tree model with specified columns of intrest using default hyperparameter. It measures preformance of the decision tree model with training accuracy and cross validation accuracy. Next, it adjusts the decision tree model hyperparameter and determines the best hyperparameter while also providing a graphic. The user should then reconstruct the model with the best hyperparameter and measure the performance with training, cross valdiation, testing accuracy, and confusion matrix. 
     Then it constructs knn model with specified columns of intrest using default hyperparameter. It measures preformance of the knn model with training accuracy and cross validation accuracy. Next, it adjusts the knn model hyperparameter and determines the best hyperparameter while also providing a graphic. The user should then reconstruct the model with the best hyperparameter and measure the performance with training, cross valdiation, testing accuracy, and confusion matrix.  
      
      
   


Scope and Limitations: There are no ethical concerns, but there are some limitations to the data. The Palmer Penguins dataset only contains data on 3 species of penguins. If we wanted to learn more about all species of penguins, this dataset would not be enough to answer all of our questions. A possible extension using a dataset with more data from all species of penguins would allow us to visualize data on all species of penguins. We are also only using two different models so we might be limited in performance. 

References and Acknowledgement: References to PIC16A Fall 22 taught by Harlin.

The dataset used is Palmer Penguins. This is a dataset containing various measurements of 3 different species of penguins collected in the Palmer Archipelago by Dr. Kristen Gorman. We used to .csv file found on Canvas.

Links to Tutorials: NA 
Software Demo Video: NA 
