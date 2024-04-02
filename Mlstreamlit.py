import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st

df=pd.read_csv(r"D:\Users\Admin\Desktop\datasciene\DemoMongoDB\phonepe\diabetes_prediction_dataset.csv")
encode=OrdinalEncoder()
df["gender"]=encode.fit_transform(df[["gender"]])
encode1=OrdinalEncoder()
df["smoking_history"]=encode1.fit_transform(df[["smoking_history"]])

x=df.drop("diabetes",axis=1)
y=df["diabetes"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier().fit(x_train,y_train)


gender = {'Female':0, 'Male':1, 'Other':2}
smoking_history = {'never':4, 'No Info':0, 'current':1, 'former':3, 'ever':2, 'not current':5}
yes_no = {"No": 0, "Yes": 1}

st.title("Diabetes Test Predictor")
st.write("welcome to Diabetes prediction application")
st.header("Please fill the form to see result")

col=st.columns((1.5,1.5),gap='medium') 
with col[0]:
    gender_sele=st.selectbox("choose your gender",{"Female":0,"Male":1,"Other":2})
    age=st.number_input("enter your age",min_value =0, max_value =100, value=None)
    hypertension=st.selectbox("Do you have hypertension?",{"No": 0, "Yes": 1})
    heartdisease=st.selectbox("Do you have heartdisease?",{"No": 0, "Yes": 1})
    Smoking=st.selectbox("Do you smoke",{'never':4, 'No Info':0, 'current':1, 'former':3, 'ever':2, 'not current':5})
    bmi = st.number_input("Enter your BMI", min_value=0, max_value=100,value=None )
    HbA1c_level = st.number_input("Enter your HbA1c level", min_value=0.0, max_value=100.0, value=None )
    blood_glucose_level = st.number_input("Enter your blood glucose level", min_value=0.0, max_value=1000.0, value=None )
    input=np.array([[gender[gender_sele],age,yes_no[hypertension],yes_no[heartdisease],smoking_history[Smoking],bmi,HbA1c_level,blood_glucose_level]])

diab=st.button("check_diabetes")
if diab:
    pred=model.predict(input)
    if pred==1:
        st.error("You Have Diabetes")
    else:
        st.success("You dont have Diabetes")
        
    