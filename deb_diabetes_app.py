#!/usr/bin/env python
# coding: utf-8

# In[1]:

#front end UI development using streamlit
#Debapriyo Paul

#okay so now we have build the ml model - after that we want to make this an interactive app that folks can easily use
import streamlit as st
import joblib
import pandas as pd
from PIL import Image


# In[2]:

#try to setup
st.set_page_config(page_title='Deb Diabetes Predictor App',
                   page_icon='https://smallimg.pngkey.com/png/small/43-433372_image-for-free-caduceus-medical-symbol-health-high.png',
                   layout="centered",
                   initial_sidebar_state='expanded')


@st.cache(allow_output_mutation=True)
def load(scaler_path, model_path):
    sc = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return sc , model
#define a function that can load in the scalar and the model that we created in the previous notebook


# In[3]:


def inference(row, scaler, model, feat_cols):
    df = pd.DataFrame([row], columns = feat_cols)
    X = scaler.transform(df)
    features = pd.DataFrame(X, columns = feat_cols)
    if (model.predict(features)==0):
        return "This is a healthy person!"
    else: return "This person has high chances of having diabetics!"
#define a function that can input the variables, run the model and output the result


# In[ ]:


#create the app with streamlit
st.title('Diabetes Prediction App')
st.write('The data for the following example is originally from the National Institute of Diabetes and Digestive and Kidney Diseases and contains information on females at least 21 years old of Pima Indian heritage. This is a sample application and cannot be used as a substitute for real medical advice.')
image = Image.open('data/diabetes_image.jpg')
st.image(image, use_column_width=True)
st.write('Please fill in the details of the person under consideration in the left sidebar and click on the button below!')


age =           st.sidebar.number_input("Age in Years", 1, 150, 25, 1)
pregnancies =   st.sidebar.number_input("Number of Pregnancies", 0, 20, 0, 1)
glucose =       st.sidebar.slider("Glucose Level", 0, 200, 25, 1)
skinthickness = st.sidebar.slider("Skin Thickness", 0, 99, 20, 1)
bloodpressure = st.sidebar.slider('Blood Pressure', 0, 122, 69, 1)
insulin =       st.sidebar.slider("Insulin", 0, 846, 79, 1)
bmi =           st.sidebar.slider("BMI", 0.0, 67.1, 31.4, 0.1)
dpf =           st.sidebar.slider("Diabetics Pedigree Function", 0.000, 2.420, 0.471, 0.001)
#define how you want to display the predictor input areas/types with bounds and default values


# In[ ]:


row = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]

if (st.button('Find Health Status')):
    feat_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    sc, model = load('model/scaler.joblib', 'model/model.joblib')
    result = inference(row, sc, model, feat_cols)
    st.write(result)
#create a button, which when clicked calls the functions created above to load in the model, apply the predictor values and 
#give out a result which is then printed
    

