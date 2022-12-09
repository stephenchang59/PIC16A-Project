# PIC16A-Project
Penguins Project

Stephen, Glenda, Armine, and Ryan

For our group project, we decided to do the clustering penguin analysis. We will work with a dataset containing features such as sex, island of origin, species, flipper length, etc. for over 300 penguins. Our goal will be to construct two different models using the KNN algorithm and decision tree/random forest algorithm and compare their performance on predicting the species of penguins. First, we will do EDA to show which features are correlated to our response variable, see the distribution of our data, and check cross-feature correlation. Next, we will use cross-validation to select the best hyperparameters. Finally, we will display our results which will be their accuracy scores on testing data. 

Python Packages Used: matplotlib 3.4.3, sklearn 1.1.3, numpy 1.20.3, pandas 1.3.4, seaborn 0.11.2

Description of Demo File/Walk through:
\
We wanted to give the user flexibility in selecting which feature to predict and which features to include in the model. 
\
\
    First, the demo file cleans the data and preforms EDA using the custom function with the user specified columns of interest. 
    ![image](https://user-images.githubusercontent.com/103079590/206025695-2bbc06bb-ee95-4c35-bd60-c17bf0b8d747.png)
    ![image](https://user-images.githubusercontent.com/103079590/206026083-12c749b7-dd70-4c12-98f2-df4c40aeb694.png)
    ![image](https://user-images.githubusercontent.com/103079590/206026132-a2e89a32-5972-43a6-9232-ae99f3e5464b.png)
    \
    Then it constructs decision tree model with specified columns of intrest using default hyperparameter.
    \
    \
    DT = classifier("Decision Tree", train, test, ["Culmen Length (mm)", "Island"], "Species")
    \
    \
    In this case we used culmen length and island to predict species. It measures preformance of the decision tree model with training accuracy and cross validation accuracy. Next, it adjusts the decision tree model hyperparameter and determines the best hyperparameter while also providing a graphic for reference.
    \
    ![image](https://user-images.githubusercontent.com/103079590/206026253-a39f7f10-8778-41cc-8925-ee10c2a91411.png)
The user should then reconstruct the model with the best hyperparameter and measure the performance with training, cross valdiation, testing accuracy, and confusion matrix. 
    \
    \
    DT = classifier("Decision Tree", train, test, ["Culmen Length (mm)", "Island"], "Species", hyperparameter_decision_tree  = 3)
    \
    \
    ![image](https://user-images.githubusercontent.com/103079590/206028239-4fc04c64-20fa-4628-8017-8a6ae2ba5f69.png)
    Then it constructs knn model with specified columns of intrest using default hyperparameter.
    \
    \
    knn = classifier("Knn", train, test, ["Flipper Length (mm)", "Body Mass (g)"], "Species")
    \
    \
    In this case we used flipper length and body mass. It measures preformance of the knn model with training accuracy and cross validation accuracy. Next, it adjusts the knn model hyperparameter and determines the best hyperparameter while also providing a graphic. 
    ![image](https://user-images.githubusercontent.com/103079590/206026518-68d22a68-c985-4154-a9d6-f0e13029fc10.png)
    The user should then reconstruct the model with the best hyperparameter and measure the performance with training, cross valdiation, testing accuracy, and confusion matrix.  
    \
    \
    knn = classifier("Knn", train, test, ["Flipper Length (mm)", "Body Mass (g)"], "Species", hyperparameter_knn = 13)
    \
    \
      ![image](https://user-images.githubusercontent.com/103079590/206026591-91e36193-3c56-4c45-b29d-24650ee9ec0a.png)

      
   


Scope and Limitations: There are no ethical concerns, but there are some limitations to the data. The Palmer Penguins dataset only contains data on 3 species of penguins. If we wanted to learn more about all species of penguins, this dataset would not be enough to answer all of our questions. A possible extension using a dataset with more data from all species of penguins would allow us to visualize data on all species of penguins. We are also only using two different models so we might be limited in performance. It is also not that representative if we want our algorithm to work well on other unseen data sets.

References and Acknowledgement: References to PIC16A Fall 22 taught by Harlin.

The dataset used is Palmer Penguins. This is a dataset containing various measurements of 3 different species of penguins collected in the Palmer Archipelago by Dr. Kristen Gorman. We used to .csv file found on Canvas.

Links to Tutorials: NA 
Software Demo Video: NA 
