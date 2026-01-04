import streamlit as st
import pandas as pd
import joblib
import os
import sys
from datetime import datetime

from src.exception import CustomeException
from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler
from src.components.data_ingestion import DataIngestion
from src.pipelines.train_pipeline import run_training_pipeline


st.set_page_config(page_title="Student Marks Predictor", page_icon=":pencil:", layout="wide")

# Top-level title (kept for SEO / browser tab)
st.title("📘 Student Math Score Predictor")

# Sidebar navigation for Home / Predict
page = st.sidebar.selectbox("Go to", ["Home", "Predict", "Complete data"])

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
    st.markdown("""
    Welcome to the **Student Performance Prediction App** - a tool that uses machine learning to estimate a student's **Mathematics score** based on a small set of academic and demographic features.

    ---

    ## 🎯 About This Project
    This project predicts a student's **Math score** using both **categorical** and **numerical** features:

    ### 🔹 Features Used
    - **Gender**  
    - **Race/Ethnicity**  
    - **Parental Level of Education**  
    - **Lunch Type**  
    - **Test Preparation Course**  
    - **Reading Score**  
    - **Writing Score**

    The model is trained as a **regression model**.  
    The data is stored in a **local SQL database**, and every time you retrain, the system:

    1. Loads all data from SQL  
    2. Runs the full ingestion + transformation pipeline + training pipeline
    3. Trains a new model  
    4. Saves updated model weights  

    ---

    ## 🔄 Training the Model
    To train the model with the **latest data**, simply click:

    ### 👉 ***“Load and Train with new data”***

    This will rebuild the model using all data added since the last training.

    Once training is complete, switch to the **Predict** page from the left sidebar.

    ---

    ## 🔍 Making Predictions
    On the **Predict** page:

    1. Choose values for all categorical features  
    2. Enter the student's **Reading** and **Writing** scores  
    3. Click **Predict**  
    4. View the model’s estimated **Math score**

    ---

    ## 📊 View Complete Dataset
    If you want to explore all stored data, select **“Complete Data”** from the left sidebar.

    ---

    ## 🧠 Why These Features?
    Reading and writing scores are often strong indicators of overall academic performance.  
    The demographic and background features help the model understand additional context that may influence learning outcomes.

    ---

    ## 🧩 View the Source Code
    Curious about how everything works behind the scenes?  
    You can explore the full project on GitHub:

    ### 🔗 **GitHub Repository:**  
    [Student Marks Predictor](https://github.com/akshaypratap008/Student-Marks-Predictor)

    ---

    Enjoy exploring the model and experimenting with it!
    """)

    btn = st.button('Load and Train with new data')
    last_training_datetime = f'Last trained at {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    if btn:
        with st.spinner('Training Model with latest data...'):
            result = run_training_pipeline()
        st.success("Training Completed!")
        st.write('Make predictions based on new data now')
        last_training_datetime = f'Last trained at {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    st.caption(last_training_datetime)
    st.caption("[LinkedIn](https://www.linkedin.com/in/akshaypratap08/)", text_alignment='right')
    st.caption("[Github](https://github.com/akshaypratap008)", text_alignment='right')
    st.caption("Created by Akshay :)", text_alignment='right')

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
    btn = col2.button("Predict Math's Score", use_container_width=True)
    if btn:
        if int(result) >= 50:
            st.balloons()
        st.markdown(f"<h2 style='text-align:center; color:white;'>The Student's Predicted Math Score", unsafe_allow_html=True, text_alignment='justify')

        st.markdown(
        f"""
        <div style="
            background-color:#1E1E1E;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
            margin-top: 20px;
        ">
            <h1 style="color:white; font-size: 40px; margin: 0;">
                🎯 {int(result)}
            </h1>
            <p style="color:#CCCCCC; font-size: 20px; margin-top: 10px;">
                Predicted Math Score
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

        

def show_full_data():
    obj = DataIngestion(
        host = '127.0.0.1',
        user = 'root',
        password = '',
        database= 'mlproject1'
    )
    query = 'SELECT * FROM stud'
    st.table(obj.load_data(query=query))
        
if page == 'Home':
    show_home()
elif page == 'Predict':
    show_predict()
elif page == 'Complete data':
    show_full_data()



