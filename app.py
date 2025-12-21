import streamlit as st
import pandas as pd
import joblib
import os
import sys

from src.exception import CustomeException
from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Marks Predictor", page_icon=":pencil:", layout="centered")

# Top-level title (kept for SEO / browser tab)
st.title("Student Marks Predictor")

# Sidebar navigation for Home / Predict
page = st.sidebar.selectbox("Go to", ["Home", "Predict"])

# Shared constants
GENDERS = ["female", "male"]
RACES = ["group A", "group B", "group C", "group D", "group E"]
PARENT_EDUCATION = [
    "Some College",
    "Associate's Degree",
    "High School",
    "Some High School",
    "Bachelor's Degree",
    "Master's Degree",
]
LUNCH = ["standard", "free/reduced"]
TEST_PREP = ["none", "completed"]

def show_home():
    st.header("About this project")
    st.write(
        "This project predicts a student's Mathematics score using a small set of features:"
    )
    st.markdown(
        """
- Categorical features: Gender, Race/Ethnicity, Parental Level of Education, Lunch type, Test Preparation Course  
- Continuous features: Reading score, Writing score

The model is trained as a regression model (saved and loaded from your project's pipeline) and exposed via this Streamlit app.  
Use the `Predict` page to provide inputs and get a predicted Mathematics score.
"""
    )

    st.subheader("Why these features?")
    st.write(
        "Reading and writing scores are strong predictors of overall performance in mathematics in many datasets. "
        "The demographic and background categorical features help the model capture additional variance related to student context."
    )

    st.subheader("How to use")
    st.markdown(
        """
1. Go to the Predict page using the left sidebar.  
2. Choose values for the categorical features and enter reading & writing scores.  
3. Click the Predict button to get the model's predicted Math score.
"""
    )

    st.subheader("Model / Data notes")
    st.write(
        "This app expects a saved scikit-learn pipeline (or an object with a `.predict()` method) at `model.pkl` in the project root. "
        "The pipeline should accept a DataFrame with columns (exact names):"
    )
    st.code(
        "['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'reading_score', 'writing_score']"
    )
    st.write(
        "If the pipeline contains both preprocessing and the final regressor, the app will call `model.predict(df)` directly. "
        "If your model uses different column names or expects different preprocessing, update the pipeline or this app accordingly."
    )

    st.subheader("Troubleshooting")
    st.markdown(
        """
- If you see a message that the model file is missing: save your pipeline as `model.pkl` beside this `app.py`.  
- If prediction fails with an error, check that the pipeline can accept a pandas DataFrame with the columns listed above.  
"""
    )

    st.caption("Created by the project team — adjust the text here to provide more project-specific details.")

def show_predict():
    st.sidebar.header("Select Features")
    st.sidebar.write('Choose value')
    gender = st.sidebar.selectbox("Gender", GENDERS)
    race_ethnicity = st.sidebar.selectbox("Race / Ethnicity", RACES)
    parental_level_of_education = st.sidebar.selectbox("Parental Level of Education", PARENT_EDUCATION)
    lunch = st.sidebar.selectbox("Lunch", LUNCH)
    test_preparation_course = st.sidebar.selectbox("Test Preparation Course", TEST_PREP)

    reading_score = st.sidebar.number_input("Reading score", min_value=0, max_value=100, step = 1)
    writing_score = st.sidebar.number_input("Writing score", min_value=0, max_value=100, step = 1)



    # load input data from users
    input_data = CustomData(gender.lower(), race_ethnicity, parental_level_of_education.lower(), lunch.lower(), test_preparation_course, reading_score,writing_score)

    #convert input features into a dataframe
    input_df = CustomData.get_data_as_dataframe(input_data)

    # display user selected data in dataframe 
    st.subheader("Input")
    st.table(input_df.rename(columns = {
        'gender': 'Gender',
        'race_ethnicity': 'Race/Ethnicity',
        'parental_level_of_education': "Parent's Education",
        'lunch': 'Lunch Type',
        'test_preparation_course': 'Test Preparation Course',
        'reading_score': 'Reading Score',
        'writing_score': 'Writing Score'
    }))

    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(input_df)


    col1, col2, col3= st.columns(3)
    btn = col2.button("Predict Math's Score")
    if btn:
        col2.markdown(f"This student's predicted Math's score is: {int(result)}")
        if int(result) >= 50:
            st.balloons()
        
if page == "Home":
    show_home()
else:
    show_predict()


