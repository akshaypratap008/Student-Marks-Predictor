### END TO END PROJECT

About this project
This project predicts a student's Mathematics score using a small set of features:

Features that predict the students marks are categorical features like Gender, Race/Ethnicity, Parental Level of Education, Lunch type, Test Preparation Course and numerical features like Reading score, Writing score

The model is trained as a regression model. You can use the button "Load and Train with new data" if you want to train the model on new data added since the last training. The data is stored in SQL server on my local machine. Everytime, new training happens, complete data is sent thorigh data ingestion and training pipeline and stores new model with new weights. 

Once the data is trained, switch to 'Predict' page by selecting from the dropdown on left panel. 

if you want to view the full data, switch to "Complete Data" from the left panel.

Why these features?
Reading and writing scores are strong predictors of overall performance in mathematics in many datasets. The demographic and background categorical features help the model capture additional variance related to student context.

How to use
Train the model with new data by clicking on "Load and Train with new data" button at the bottom of the page. 
Switch to predict page from the left panel. Choose values for the categorical features and enter reading & writing scores.
Click the Predict button to get the model's predicted Math score.





