import streamlit as st
import joblib
from sklearn import preprocessing

#loading trained model
classifier = joblib.load("iris_classifier.pkl")
x_scaler = joblib.load("xtrain_scale.pkl")

def prediction(s_len, s_width, p_len, p_width):
    if s_len == 0 and s_width == 0 and p_len == 0 and p_width == 0 :
        return "Insert anything"
    else:
        x_test = x_scaler.transform([[s_len, s_width, p_len, p_width]])    
        prediction = classifier.predict(x_test)
        if prediction == 0:
            pred = 'Sentosa'
        elif prediction == 1:
            pred  = 'Versicolor'
        else:
            pred = 'Virginica'

        return "Iris Species: " + pred
        
    


st.markdown("""
Iris Classification Web App with Linear Regression
""")
s_len = st.number_input("Sepal Length:", 0.0)
s_wid = st.number_input("Sepal Width:", 0.0)
p_len = st.number_input("Petal Length:", 0.0)
p_wid = st.number_input("Petal Width:", 0.0)

if st.button("Predict"):
    pred = prediction(s_len, s_wid, p_len, p_wid)
    st.write( pred )