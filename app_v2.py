import streamlit as st
from joblib import load
import pandas as pd
# Import other necessary libraries and functions

# Load your model and any other required assets
model_pred = load('Models/random_forest_model.joblib')
mileage_scaler = load('Models/mileage_scaler.joblib')
levy_scaler = load('Models/levy_scaler.joblib')
full_pipeline = load('Models/full_pipeline.joblib')


# Define a function to preprocess input data

def st_transform(input_d):
    # Replace ' km' in Mileage
    input_d["Mileage"] = input_d["Mileage"].astype(str).str.replace(' km', '').astype(int)

    # Handle Levy
    input_d["Levy"] = input_d["Levy"].astype(str).str.replace("-", '').replace("", '0').astype(float)
    return input_d


def cus_encoder(X):
    # Encoding logic
    X['Doors'].replace(['2', '4', "Other"], [1, 2, 3], inplace=True)
    X['Wheel'].replace(['Left wheel', 'Right wheel'], [1, 2], inplace=True)
    X['Drive_wheels'].replace(['Front', '4WD', "Rear"], [1, 2, 3], inplace=True)
    X['Gear_box_type'].replace(['Automatic', 'Tiptronic', "Manual", "Variator"], [1, 2, 3, 4], inplace=True)
    X['Leather_interior'].replace(['Yes', 'No'], [1, 2], inplace=True)
    X['With_Turbo'].replace(['Yes', 'No'], [1, 2], inplace=True)

    return X


def transform_input(input_df_trans):
    input_df_trans['Mileage'] = mileage_scaler.transform(input_df_trans[['Mileage']])
    input_df_trans['Levy'] = levy_scaler.transform(input_df_trans[['Levy']])
    return input_df_trans


def transform_input_for_prediction(input_df_trans_pred):
    # Assuming input_df is a single-row DataFrame with the same structure as the training data
    transformed_data = full_pipeline.transform(input_df_trans_pred)
    return transformed_data

def preprocess_data(input_data):
    initial_transformed_df_1 = st_transform(input_data)
    initial_transformed_df_2 = cus_encoder(initial_transformed_df_1)
    initial_transformed_df_2['Levy'] = pd.to_numeric(initial_transformed_df_2['Levy'], errors='coerce')
    initial_transformed_df_2['Mileage'] = initial_transformed_df_2['Mileage'].astype('int32')
    transformed_data_df_3 = transform_input(initial_transformed_df_2)
    print(transformed_data_df_3.shape)

    # Apply the third level of transformation (encoding)
    transformed_row = transform_input_for_prediction(transformed_data_df_3)
    print(transformed_row.shape)

    return transformed_row



# Initialize session state for page navigation and user inputs
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

# Define pages
def page1():
    st.header("Model Details")
    st.session_state.user_inputs['Manufacturer'] = st.text_input('Manufacturer')
    st.session_state.user_inputs['Model'] = st.text_input('Model')
    st.session_state.user_inputs['Prod_year'] = st.number_input('Production Year', min_value=1900, max_value=2023, step=1)

def page2():
    st.header("Specifications")
    st.session_state.user_inputs['Fuel_type'] = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric', 'Hybrid', 'LPG', 'CNG', 'Other'])
    st.session_state.user_inputs['Engine_volume'] = st.text_input('Engine Volume')
    st.session_state.user_inputs['Mileage'] = st.number_input('Mileage', min_value=0)
    st.session_state.user_inputs['Cylinders'] = st.number_input('Cylinders', min_value=0, step=1)


def page3():
    st.header("Additional Features")
    st.session_state.user_inputs['Doors'] = st.selectbox('Doors', ['2', '4', 'Other'])
    st.session_state.user_inputs['Wheel'] = st.selectbox('Wheel', ['Left wheel', 'Right wheel'])
    st.session_state.user_inputs['Color'] = st.text_input('Color')
    st.session_state.user_inputs['Airbags'] = st.number_input('Airbags', min_value=0, step=1)

# Function to save inputs and move to next page
def next_page():
    st.session_state.page_number += 1

# Function to go back to the previous page
def prev_page():
    st.session_state.page_number -= 1



# Streamlit UI components
st.title('Car Price Estimation')

# Display the appropriate page
if st.session_state.page_number == 1:
    page1()
    st.button('Next', on_click=next_page)
elif st.session_state.page_number == 2:
    page2()
    st.button('Back', on_click=prev_page)
    st.button('Next', on_click=next_page)
elif st.session_state.page_number == 3:
    page3()
    st.button('Back', on_click=prev_page)
    if st.button('Predict'):
        # Gather all inputs
        input_data = pd.DataFrame([st.session_state.user_inputs], columns=['Levy', 'Manufacturer', 'Model', 'Prod_year',
                                                                           'Category', 'Leather_interior', 'Fuel_type',
                                                                           'Engine_volume', 'Mileage', 'Cylinders',
                                                                           'Gear_box_type', 'Drive_wheels', 'Doors',
                                                                           'Wheel', 'Color', 'Airbags', 'With_Turbo'])
        print(input_data)
        # Preprocess the data
        processed_data = preprocess_data(input_data)  # Make sure this function is defined as in your Flask app
        # Make a prediction
        prediction = model_pred.predict(processed_data)
        # Display the prediction
        st.write(f'Estimated Price: ${prediction[0]}')