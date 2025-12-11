import streamlit as st
import streamlit_option_menu as som
import pandas as pd

df = pd.DataFrame({"BMI VALUE":["<18.5","18.5-24.9","25-29.9","30-34.9","35-39.9","greater than 40"],
                   "Category":["Underweight","Healthy or Normal Weight","OverWeight","Obesity Class 1","Obesity Class 2","Obesity Class 3"]})
#st.header("-------------------BMI CALCULATOR----------------")
#with st.sidebar with these code we can move the sidebar in vertical positon in like sidebar
with st.sidebar:
    option_selected = som.option_menu(menu_title="Menu",options=["Welcome","About BMI","BMI Calculator","BMI Category"],   orientation="vertical")
if option_selected=="Welcome":
    st.title("Welcome To BMI Calculator")
if option_selected=="BMI Calculator":
    st.title("BMI CALCULTOR")
    weight = st.number_input("Enter Weight in Kg:-",min_value=1)
    height = st.number_input("Enter Your Height in Meters:-",min_value=1.00)
    bmi_value = weight/height**2
    if st.button("Calculate"):
        st.write("Your BMI VALUE:",bmi_value)
        if bmi_value<18.5:
            st.write("UnderWeight")
        if bmi_value>=18.5 and bmi_value<=24.9:
            st.write("Healthy or Normal Weight")
        if bmi_value>=25 and bmi_value<=29.9:
            st.write("OverWeight")
        if bmi_value>=30 and bmi_value<=34.9:
            st.write("Obesity Class 1")
        if bmi_value>=35 and bmi_value<=39.9:
            st.write("Obesity Class 2")
        if bmi_value>=40:
            st.write("Obesity Class 3")
if option_selected=="About BMI":
    st.title("ABOUT BMI")
    st.write("Body Mass Index (BMI) is a simple numerical value calculated using a person’s height and weight to determine whether they are underweight, healthy, overweight, or obese. It is widely used as a quick health indicator because it gives a general idea of body fat based on easy measurements. BMI helps in identifying potential health risks related to weight, such as heart disease, diabetes, and high blood pressure.")
    st.write("Although BMI is useful, it does not measure body fat directly and may not be accurate for athletes, children, or people with high muscle mass. It also does not consider age, gender, or body composition. Despite these limitations, BMI remains one of the most common tools for basic health assessment and is often used in fitness apps, hospitals, and health checkups.")
if option_selected=="BMI Category":
    st.title("BMI CATEGORY WEIGHT WISE")
    st.dataframe(df)
