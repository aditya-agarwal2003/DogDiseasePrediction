# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from constants import SKIN_CURES
# from keras.models import load_model
# from keras.preprocessing import image
import numpy as np
from PIL import Image


diabetes_model = pickle.load(open('dog_diabetes_model.sav','rb'))
heart_model = pickle.load(open('dog_heart_model.sav','rb'))
# skin_model = load_model('dog_skin_disease_model.h5')

diabetes_dataset = pd.read_csv('dogDiabetesData.csv')
heart_dataset = pd.read_csv('dogHeartData.csv')


# st.markdown(f'<a href="javascript:window.history.back();">Back to Landing Page</a>', unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu('Dog Disease Prediciton System',
                           ['Dog Diabetes Prediction',
                            'Dog Heart Disease Prediction'],
                          #  ['Dog Skin Disease'],
                           icons = ['activity','heart', 'upload'],
                           default_index = 0)
    

breeds = ['Yorkshire Terrier', 'Siberian Husky', 'Labrador Retriever',
          'Chihuahua', 'Great Dane', 'Bulldog', 'Golden Retriever',
          'Dachshund', 'Beagle', 'Poodle', 'Pomeranian', 'Shih Tzu',
          'German Shepherd', 'Boxer', 'Rottweiler']

# skin_class_labels = ['Bacterial_dermatosis', 'Fungal_infections', 'Healthy', 'Hypersensitivity_allergic_dermatosis']

breed_dict = {breed: index for index, breed in enumerate(sorted(breeds))}

sorted_breed_dict = dict(sorted(breed_dict.items()))

# for diabetes prediction
if selected == 'Dog Diabetes Prediction':
    
    
    st.title('Dog Diabetes Prediciton')
        # getting the input data from the user

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Glucose = st.text_input('Glucose Level')
        
    with col2:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col3:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col1:
        Insulin = st.text_input('Insulin Level')
    
    with col2:
        Polyuria = st.text_input('Polyuria value')
    
    with col3:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col1:
        Age = st.text_input('Age of the Dog')

    
    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        
        std_data = [[int(Glucose), int(BloodPressure), int(SkinThickness), int(Insulin), float(Polyuria), float(DiabetesPedigreeFunction), int(Age)]]

        diab_prediction = diabetes_model.predict(std_data)
        if diab_prediction[0] == 0:
            diab_diagnosis = 'Your Dog does not have Diabetes'
        else:
            diab_diagnosis = 'Your Dog might have Diabetes'
            
    st.success(diab_diagnosis)
        
    
if selected == 'Dog Heart Disease Prediction':
    
    st.title('Dog Heart Disease Prediction')
    
    col1, col2, col3 = st.columns(3)  
    
    with col1:
        Stroke = st.text_input('Stroke (1 for Yes, 0 for No)')
        
    with col2:
        DiffWalking = st.text_input('Difficulty in Walking (1 for Yes, 0 for No)')
    
    with col3:
        Sex = st.text_input('Gender (1 for Male, 0 for Female)')
    
    with col1:
        Diabetic = st.text_input('Diabetic (1 for Yes, 0 for No)')
    
    with col2:
        PhysicalActivity = st.text_input('Physical Activity (1 for Yes, 0 for No)')
    
    with col3:
        GenHealth = st.text_input('General Health (Range 1(poor) to 5(excellent))')
    
    with col1:
        SleepTime = st.text_input('SleepTime (In hours 0-23)')
    
    with col2:
        KidneyDisease = st.text_input('Kidney Disease(1 for Yes, 0 for No)')
    
    with col3:
        Breed = st.selectbox("Select an option:", breeds)

    with col1:
        Age = st.text_input('Age (max. 14)')
        
    
    
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Heart Disease Test Result"):
        std_data = [[int(Stroke), int(DiffWalking), int(Sex), int(Diabetic), int(PhysicalActivity),
                     int(GenHealth), int(SleepTime), int(KidneyDisease), int(Age), 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        std_data[0][9+breed_dict[Breed]] = 1
        
        heart_prediction = heart_model.predict(std_data)                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = "Your Dog might have Heart disease"
        else:
          heart_diagnosis = "Your Dog does not have Heart disease"
        
    st.success(heart_diagnosis)


# if selected == 'Dog Skin Disease':
#     uploaded_file = st.file_uploader("Choose an image...", type="jpg")

#     if uploaded_file is not None:
#         # Convert the uploaded file to an image
#         img = Image.open(uploaded_file)
#         st.image(img, caption='Uploaded Image', use_column_width=True)
        
#         # Preprocess the image
#         img = img.resize((150, 150))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0

#         # Make a prediction
#         prediction = skin_model.predict(img_array)
#         predicted_class = skin_class_labels[np.argmax(prediction)]

#         if (predicted_class == 'Bacterial_dermatosis'):
#           skin_diagnosis = "Your Dog might have Bacterial Dermatosis"
#           message = SKIN_CURES['Bacterial_dermatosis']
#         elif (predicted_class == 'Fungal_infections'):
#           skin_diagnosis = "Your Dog might have Fungal Infections"
#           message = SKIN_CURES['Fungal_infections']
#         elif (predicted_class == 'Healthy'):
#           skin_diagnosis = "Yay! Your Dog is Healthy"
#           message = SKIN_CURES['Healthy']
#         else:
#           skin_diagnosis = "Your Dog might have Hypersensitivity Allergic Dermatosis"
#           message = SKIN_CURES['Hypersensitivity_allergic_dermatosis']

#         st.success(skin_diagnosis)
#         st.success(message)
