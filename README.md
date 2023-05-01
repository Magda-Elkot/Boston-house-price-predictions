# Boston Housing Price Prediction

## Overview
  The goal of this project was to accurately predict the price of owner-occupied homes in Boston based on the features provided 
  in the Boston Housing dataset. 
  The project involved data preprocessing, feature engineering, model selection, and neural network architecture design to achieve the goal.

## Data
  The Boston Housing dataset was used for this project. The dataset includes information on various features of homes in Boston, such as 
  crime rate, number of rooms, and distance to employment centers.
  
## Data Preprocessing
  The data was preprocessed by scaling the features using StandardScaler and splitting the dataset into training and testing sets. 
  Initial data exploration was performed to check for missing values, duplicates, and correlations between the features and target variable.

## Model Selection
  Several machine learning models, including Linear Regression, Support Vector Regression (SVR), and Random Forest Regression, were trained 
  and evaluated to determine which model performed best on the task of predicting the price. 
  The results showed that the Random Forest Regression model had the highest accuracy, with an R-squared score of 0.88.

## Random Forest Regressor Results
  To better visualize the performance of the Random Forest Regressor, a bar chart was created to compare 
  the actual output values for X_test with the predicted values :
  
  ![image](https://user-images.githubusercontent.com/121414067/235501127-039dd224-c185-4255-86ec-653745463712.png)

  As you can see, the most important feature for predicting the price of a home in Boston is the average number of rooms per dwelling (RM), 
  followed by the percentage of lower status of the population (LSTAT), and the weighted distances to five Boston employment centers (DIS).

## Neural Network Model
  A neural network model was designed and trained to predict the price of homes in Boston. The neural network architecture consisted of 
  two hidden layers with 64 and 32 units, respectively, and a ReLU activation function. 
  The output layer contained a single unit with a linear activation function, which predicted the price of owner-occupied homes. 
  The model was compiled using the Adam optimizer and mean squared error loss function and trained for 188 epochs.

## Neural Network Model Results
  During the training of the neural network model, the training and validation loss were monitored to ensure that the model was not overfitting:

  As you can see, the training and validation loss decreased steadily throughout the training process, indicating that 
  the model was not overfitting to the training data.

  ![image](https://user-images.githubusercontent.com/121414067/235501641-5a14b451-61ee-4c70-bce3-29bfe39c4e0d.png)


  After training, the neural network model was evaluated on the test data using three metrics: mean absolute error, R-squared, and loss. The results showed that the     neural network outperformed the other machine learning models, achieving a mean absolute error of 2.53, R-squared of 0.90, and a loss of 10.64.

## Conclusion
  Overall, this project provided valuable experience in data preprocessing, feature engineering, model selection, and neural network architecture design. 
  By combining various machine learning techniques and selecting the best performing model, we were able to accurately 
  predict the price of owner-occupied homes in Boston, which can be useful for real estate agents, homeowners, and policy makers.

## Contact
  For more information about this project, please contact Magda El-Romany at magdaalromany@gmail.com .

## Acknowledgements
This project was completed as part of the internship program at SYNC INTERN'S.
