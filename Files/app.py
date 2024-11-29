import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

pickle_in = open(r"C:\Users\Ashith\PycharmProjects\Machiene Learning Project (Lung cancer)\SVC.pkl", "rb")
RFC = pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_note_authentication(Gender, Age, Smoking, Yellow_fingers, Anxiety, PeerPressure, Chronic_disease, Fatigue, Allergy, Wheezing, Alcohal, Coughing, Breadth_shortness, Swallowing_diff, Chest_pain):
    prediction = RFC.predict([[Gender, Age, Smoking, Yellow_fingers, Anxiety, PeerPressure, Chronic_disease,
                             Fatigue, Allergy, Wheezing, Alcohal, Coughing, Breadth_shortness, Swallowing_diff, Chest_pain]])
    print(prediction)
    return prediction


def main():
    html_temp = """
    <div style="background-color:black;padding:10px">
    <h2 style="color:white;text-align:center;">Lung Cancer App: Detecting the Signs Early </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Gender = st.text_input("Gender", "Type Here")
    Age = st.text_input("Age", "Type Here")
    Smoking = st.text_input("Smoking", "Type Here")
    Yellow_fingers = st.text_input("Yellow_fingers", "Type Here")
    Anxiety = st.text_input("Anxiety", "Type Here")
    PeerPressure = st.text_input("PeerPressure", "Type Here")
    Chronic_disease = st.text_input("Chronic_disease", "Type Here")
    Fatigue = st.text_input("Fatigue", "Type Here")
    Allergy = st.text_input("Allergy", "Type Here")
    Wheezing = st.text_input("Wheezing", "Type Here")
    Alcohal = st.text_input("Alcohal", "Type Here")
    Coughing = st.text_input("Coughing", "Type Here")
    Breadth_shortness = st.text_input("Breadth_shortness", "Type Here")
    Swallowing_diff = st.text_input("Swallowing_diff", "Type Here")
    Chest_pain = st.text_input("Chest_pain", "Type Here")

    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(Gender, Age, Smoking, Yellow_fingers, Anxiety, PeerPressure, Chronic_disease,
                                             Fatigue, Allergy, Wheezing, Alcohal, Coughing, Breadth_shortness, Swallowing_diff, Chest_pain)
    st.success('The output is {}'.format(result))

    # Sidebar options
    st.sidebar.title("Cancer Symptoms")
    st.sidebar.image("lung_cancer.jpg",
                     caption="Lung Cancer")

    # About Tab
    about_expander = st.sidebar.expander("About")
    with about_expander:
        st.write(
            "This is a Lung Cancer Detection ML App built using Streamlit.")
        st.write(
            "It predicts the likelihood of lung cancer based on various factors.")
        st.write("Built by Ashith Gowda, Vinayak, Dechamma")
        st.write(
            "GitHub Repository: https://github.com/Ashith003/Lung-Cancer-prediction-using-ML")
    # Help Tab
    help_expander = st.sidebar.expander("Help")
    with help_expander:
        st.write("This app helps in predicting lung cancer.")
        st.write(
            "Enter the required information in the main section and click 'Predict' to get the result.")
        st.write("Input for the Gender:")
        st.write("0.Female")
        st.write("1.Male")
        st.write("Input for the Symptom Features:")
        st.write("1.No")
        st.write("2.Yes")


if __name__ == '__main__':
    main()
