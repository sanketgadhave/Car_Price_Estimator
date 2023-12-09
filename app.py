# Import other necessary libraries and functions
import numpy as np
import streamlit as st
from joblib import load
import pandas as pd
import time
import matplotlib.pyplot as plt

# Load your model and any other required assets
model_pred = load('Models/gradient_boosting_model.joblib')
mileage_scaler = load('Models/mileage_scaler.joblib')
levy_scaler = load('Models/levy_scaler.joblib')
full_pipeline = load('Models/full_pipeline.joblib')

st.set_option('deprecation.showPyplotGlobalUse', False)

## Define a functions to preprocess input data

# Function to preprocess Mileage and Levy column
def st_transform(input_d):
    input_d["Mileage"] = input_d["Mileage"].astype(str).str.replace(' km', '').astype(int)
    input_d["Levy"] = input_d["Levy"].astype(str).str.replace("-", '').replace("", '0').astype(float)
    return input_d

# Function to encode few features
def cus_encoder(X):
    # Encoding logic
    X['Doors'].replace(['2', '4', "Other"], [1, 2, 3], inplace=True)
    X['Wheel'].replace(['Left wheel', 'Right wheel'], [1, 2], inplace=True)
    X['Drive_wheels'].replace(['Front', '4WD', "Rear"], [1, 2, 3], inplace=True)
    X['Gear_box_type'].replace(['Automatic', 'Tiptronic', "Manual", "Variator"], [1, 2, 3, 4], inplace=True)
    X['Leather_interior'].replace(['Yes', 'No'], [1, 2], inplace=True)
    X['With_Turbo'].replace(['Yes', 'No'], [1, 2], inplace=True)

    return X

# Function to scale Mileage and Levy
def transform_input(input_df_trans):
    input_df_trans['Mileage'] = mileage_scaler.transform(input_df_trans[['Mileage']])
    input_df_trans['Levy'] = levy_scaler.transform(input_df_trans[['Levy']])
    return input_df_trans


# Function to perform Column Transform
def transform_input_for_prediction(input_df_trans_pred):
    print(input_df_trans_pred.shape)
    transformed_data = full_pipeline.transform(input_df_trans_pred)
    return transformed_data

#Preprocessin Function to call all the preprocessin taks defined
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


# User defined functions for input field validation:
def map_to_existing_category(user_input_category):
    # Defining a mapping from user input "Other" to your existing categories
    category_mapping = {
        "Other": "Sedan",
    }

    # Checking if the user input category is in the mapping
    if user_input_category in category_mapping:
        return category_mapping[user_input_category]
    else:
        return user_input_category

def map_to_existing_fuel_type(user_input_fuel):
    # Defining a mapping from user input "Other" to your existing categories
    fuel_mapping = {
        "Other": "Petrol",
    }

    # Checking if the user input category is in the mapping
    if user_input_fuel in fuel_mapping:
        return fuel_mapping[user_input_fuel]
    else:
        return user_input_fuel

def validate_engine_volume(input_text):
    try:
        value = float(input_text)
        if 0.6 <= value <= 6.0:
            return value
        else:
            st.warning("Please enter a valid engine volume.\n"
                       "Smaller Cars: 0.6 - 2.0\n"
                       "Mid-Range Cars: 2.5 - 3.5\n"
                       "High-Performance or Sports Cars: 4.0 - 6.0")
            return st.session_state.user_inputs.get('Engine_volume', 0.6)  # Default value if invalid
    except ValueError:
        st.warning("Please enter a valid numeric value.")
        return st.session_state.user_inputs.get('Engine_volume', 0.6)  # Default value if invalid

def validate_cylinders(input_text):
    try:
        value = int(input_text)
        if value > 0 and value <= 16:  # Set the valid range for cylinders (0 exclusive, 16 inclusive)
            return value
        else:
            st.warning("Please enter a valid number of cylinders between 1 and 16.")
            return st.session_state.user_inputs.get('Cylinders', 1)  # Default value if invalid
    except ValueError:
        st.warning("Please enter a valid numeric value for cylinders.")
        return st.session_state.user_inputs.get('Cylinders', 1)  # Default value if invalid


def validate_airbags(input_text):
    try:
        value = int(input_text)
        if 0 <= value <= 10:  # Set the valid range for airbags (0 to 10)
            return value
        else:
            st.warning("Please enter a valid number of airbags between 0 and 10.")
            return st.session_state.user_inputs.get('Airbags', 0)  # Default value if invalid
    except ValueError:
        st.warning("Please enter a valid numeric value for airbags.")
        return st.session_state.user_inputs.get('Airbags', 0)  # Default value if invalid

# Initializing session state for page navigation and user inputs
if 'page_number' not in st.session_state:
    st.session_state.page_number = 1

if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

if 'report_page' not in st.session_state:
    st.session_state.report_page = False

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False


# Define pages
def page1():
    st.header("Car Model Details")
    st.session_state.user_inputs['Levy'] = st.number_input('Levy', min_value=0, value=st.session_state.user_inputs.get('Levy', 0), step=1)

    if 'other_manufacturer_input' not in st.session_state:
        st.session_state.other_manufacturer_input = ""

    with open('meta/Manufacturers.txt', 'r') as file:
        manufacturers_list = file.readlines()

    manufacturers_list = [manufacturer.strip() for manufacturer in manufacturers_list]
    default_manufacturer = st.session_state.user_inputs.get('Manufacturer_Display', "")
    index = manufacturers_list.index(
        default_manufacturer) if default_manufacturer and default_manufacturer in manufacturers_list else None

    selected_manufacturer = st.selectbox('Manufacturer', manufacturers_list, index=index, key='ManufacturerSelect')

    # Handling 'OTHER' selection
    if selected_manufacturer == 'OTHER':
        entered_manufacturer = st.text_input('Enter Manufacturer:', value=st.session_state.other_manufacturer_input,
                                             key='OtherManufacturerInput').upper()
        st.session_state.other_manufacturer_input = entered_manufacturer

        if entered_manufacturer:
            st.session_state.user_inputs['Manufacturer_Display'] = entered_manufacturer
            st.session_state.user_inputs['Manufacturer'] = 'BMW'
        else:
            st.error("Please enter a Manufacturer name.")
            return  # This will stop executing the rest of the page if no manufacturer is entered
    else:
        st.session_state.user_inputs['Manufacturer'] = selected_manufacturer
        st.session_state.user_inputs['Manufacturer_Display'] = selected_manufacturer

    if st.session_state.user_inputs['Manufacturer'] is not None and not st.session_state.user_inputs['Manufacturer'].strip():
        st.error("Please enter a Manufacturer.")

    print("Before model: ", st.session_state.user_inputs['Manufacturer'])

    st.session_state.user_inputs['Model'] = st.text_input('Model', value=st.session_state.user_inputs.get('Model', ''))
    print("After model: ", st.session_state.user_inputs['Manufacturer'])
    st.session_state.user_inputs['Prod_year'] = st.slider('Production Year', min_value=1900, max_value=2023, value=st.session_state.user_inputs.get('Prod_year', 2023))

    with open('meta/category.txt', 'r') as file:
        category_list = file.readlines()
    category_list = [category.strip() for category in category_list]
    default_category = st.session_state.user_inputs.get('Category', "")
    if default_category not in category_list:
        default_category = category_list[0]

    index = category_list.index(default_category)
    st.session_state.user_inputs['Category'] = st.selectbox('Category', category_list, index=index)
    st.session_state.user_inputs['Category'] = map_to_existing_category(st.session_state.user_inputs['Category'])
    if (st.session_state.user_inputs['Manufacturer'] is not None and st.session_state.user_inputs[
        'Manufacturer'].strip() == '') or \
            st.session_state.user_inputs['Model'].strip() == '' or st.session_state.user_inputs['Prod_year'] == 0:
        st.warning("Please fill in all required fields before proceeding.")
        st.session_state.warning_displayed = True
        st.stop()



def page2():
    st.header("Specifications")
    st.session_state.user_inputs['Leather_interior'] = st.selectbox('Leather Interior', ['Yes', 'No'], index=0 if st.session_state.user_inputs.get('Leather_interior', 'Yes') == 'Yes' else 1)
    with open('meta/fuel_type.txt', 'r') as file:
        fuel_list = file.readlines()
    fuel_list = [fuel.strip() for fuel in fuel_list]
    default_fuel_type = st.session_state.user_inputs.get('Fuel_type', "")
    if default_fuel_type not in fuel_list:
        default_fuel_type = fuel_list[0]

    index = fuel_list.index(default_fuel_type)
    st.session_state.user_inputs['Fuel_type'] = st.selectbox('Fuel_type', fuel_list, index=index)
    st.session_state.user_inputs['Fuel_type'] = map_to_existing_fuel_type(st.session_state.user_inputs['Fuel_type'])
    user_input = st.text_input('Engine Volume', value=str(st.session_state.user_inputs.get('Engine_volume', 0.6)))
    st.session_state.user_inputs['Engine_volume'] = validate_engine_volume(user_input)
    if 'Mileage' in st.session_state.user_inputs:
        mileage = st.session_state.user_inputs['Mileage']
    else:
        mileage = 0
    st.session_state.user_inputs['Mileage']  = st.number_input('Mileage', min_value=0, value=mileage)

    if 'Cylinders' not in st.session_state.user_inputs:
        st.session_state.user_inputs['Cylinders'] = 1
    user_input_cylinders = st.text_input('Cylinders', value=str(st.session_state.user_inputs.get('Cylinders', 0)))
    st.session_state.user_inputs['Cylinders'] = validate_cylinders(user_input_cylinders)
    if st.session_state.user_inputs['Engine_volume'] < 0.6 or not (1 <= st.session_state.user_inputs['Cylinders'] <= 16):
        st.warning("Please fill in all required fields before proceeding.")
        st.session_state.warning_displayed = True
        st.stop()



def page3():
    st.header("Additional Features")
    st.session_state.user_inputs['Gear_box_type'] = st.selectbox('Gear Box Type', ['Automatic', 'Manual', 'Tiptronic', 'Variator'], index=0 if st.session_state.user_inputs.get( 'Gear_box_type', 'Automatic') == 'Automatic' else 3)
    st.session_state.user_inputs['Drive_wheels'] = st.selectbox('Drive Wheels', ['Front', 'Rear', '4WD'], index=0 if st.session_state.user_inputs.get( 'Drive_wheels', 'Front') == 'Front' else 2)
    st.session_state.user_inputs['Doors'] = st.selectbox('Doors', ['2', '4'], index=0 if st.session_state.user_inputs.get('Doors', '2') == '2' else 2)
    st.session_state.user_inputs['Wheel'] = st.selectbox('Wheel', ['Left wheel', 'Right wheel'], index=0 if st.session_state.user_inputs.get('Wheel', 'Left wheel') == 'Left wheel' else 1)
    st.session_state.user_inputs['Color'] = st.text_input('Color')
    with open('meta/colors.txt', 'r') as file:
        color_list = file.readlines()
    color_list = [color.strip() for color in color_list]
    print("CHECKING COLOR")
    if st.session_state.user_inputs['Color'] not in color_list:
        print("IN IFFFFFFFFF")
        st.session_state.user_inputs['Color'] = 'Red'

    color_value = st.session_state.user_inputs['Color']
    if color_value in color_list or color_value == 'Red':
        st.session_state.user_inputs['Color'] = color_value

    user_input_airbags = st.text_input('Airbags', value=str(st.session_state.user_inputs.get('Airbags', 0)))
    st.session_state.user_inputs['Airbags'] = validate_airbags(user_input_airbags)
    st.session_state.user_inputs['With_Turbo'] = st.selectbox('With Turbo', ['Yes', 'No'], index=0 if st.session_state.user_inputs.get('With_Turbo', 'Yes') == 'Yes' else 1)

    required_fields = ['Levy', 'Manufacturer', 'Model', 'Prod_year', 'Category', 'Leather_interior', 'Fuel_type',
                       'Engine_volume', 'Mileage', 'Cylinders', 'Gear_box_type', 'Drive_wheels', 'Doors', 'Wheel',
                       'Color', 'Airbags', 'With_Turbo']
    for field in required_fields:
        if st.session_state.user_inputs[field] == '' or st.session_state.user_inputs[field] == 0:
            st.warning("Please fill in all required fields before making a prediction.")
            st.session_state.warning_displayed = True
            st.stop()

st.set_page_config(layout="wide")

# Function to save inputs and move to next page
def next_page():
    st.session_state.page_number += 1

# Function to go back to the previous page
def prev_page():
    st.session_state.page_number -= 1

# Function to go to report page and generate report
def report_page():
    st.header("Prediction Report")
    st.markdown("### User Input Summary")
    user_input_items = list(st.session_state.user_inputs.items())
    for i in range(0, len(user_input_items), 2):
        col1, col2 = st.columns(2)
        key, value = user_input_items[i]
        with col1:
            st.markdown(f"""
                <style>
                .report-box {{
                    border: 2px solid #dedede;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px 0;
                }}
                </style>
                <div class="report-box">
                    <h4>{key.replace('_', ' ').title()}</h4>
                    <p>{value}</p>
                </div>
                """, unsafe_allow_html=True)

        if i + 1 < len(user_input_items):
            key, value = user_input_items[i + 1]
            with col2:
                st.markdown(f"""
                    <style>
                    .report-box {{
                        border: 2px solid #dedede;
                        border-radius: 5px;
                        padding: 10px;
                        margin: 10px 0;
                    }}
                    </style>
                    <div class="report-box">
                        <h4>{key.replace('_', ' ').title()}</h4>
                        <p>{value}</p>
                    </div>
                    """, unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:green;'>Estimated Price: ${st.session_state.estimated_price}</h3>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Factors affecting price</h3>", unsafe_allow_html=True)
    with st.expander("Price Estimation Over Different Production Years", expanded=False):
        plot_price_over_time(st.session_state.input_data_cpy, model_pred, preprocess_data)
    with st.expander("Average Estimated Price by Category", expanded=False):
        plot_category_comparison(st.session_state.input_data_cpy, model_pred, preprocess_data)

# Plot function for category comparision
def plot_category_comparison(input_data_category, model, preprocess_function):
    category_list = ['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine', 'Pickup']
    average_prices = []

    for cat in category_list:
        temp_data = input_data_category.copy()
        temp_data['Category'] = cat
        processed_data = preprocess_function(temp_data)

        prediction = model.predict(processed_data)
        average_prices.append(np.mean(prediction))

    # Plotting
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(category_list, average_prices, color='skyblue')
    ax.set_title('Average Estimated Price by Fuel Type')
    ax.set_xlabel('Category')
    ax.set_ylabel('Average Estimated Price')
    st.pyplot(fig)

# Plot function for price over time
def plot_price_over_time(input_data_time, model, preprocess_function):
    years = np.arange(2000, 2024)
    prices = []

    for year in years:
        temp_data = input_data_time.copy()
        temp_data['Prod_year'] = year
        processed_data = preprocess_function(temp_data)
        prediction = model.predict(processed_data)
        prices.append(prediction[0])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(years, prices, marker='o')
    plt.title('Estimated Car Price Over Different Production Years')
    plt.xlabel('Production Year')
    plt.ylabel('Estimated Price')
    st.pyplot()


# Streamlit UI components
st.markdown("""
    <style>
    .heading-bar {
        color: white;
        background-color: #D14A2C; 
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }.heading-bar h1 {
        color: white; /* Specifically target h1 tag */
        margin: 0; /* Optional: Removes default margin */
    }
    </style>
    <div class='heading-bar'>
        <h1>Car Price Estimator</h1>
    </div>
    """, unsafe_allow_html=True)

# Displaying the appropriate page
if st.session_state.report_page:
    report_page()
    if st.button('Back to Prediction'):
        st.session_state.report_page = False
        st.experimental_rerun()
else:
    if st.session_state.page_number == 1:
        page1()
        next_col, _ = st.columns([1, 9])
        with next_col:
            st.button('Next', on_click=next_page)

    elif st.session_state.page_number == 2:
        page2()
        back_col, next_col = st.columns(2)
        with back_col:
            st.button('Back', on_click=prev_page)

        with next_col:
            st.button('Next', on_click=next_page)

    elif st.session_state.page_number == 3:
        page3()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('Back'):
                st.session_state.page_number -= 1
        with col2:
            prediction_button = st.button('Predict')

        with col3:
            report_button_enabled = not st.session_state.prediction_made
            report_button = st.button('Generate Report', disabled=report_button_enabled)

        prediction_placeholder = st.empty()
        if 'success_message' in st.session_state:
            st.success(st.session_state.success_message)

        if prediction_button:
            with st.spinner('Predicting...'):
                input_data = pd.DataFrame([st.session_state.user_inputs], columns=['Levy', 'Manufacturer', 'Model', 'Prod_year',
                                                                                   'Category', 'Leather_interior', 'Fuel_type',
                                                                                   'Engine_volume', 'Mileage', 'Cylinders',
                                                                                   'Gear_box_type', 'Drive_wheels', 'Doors',
                                                                                   'Wheel', 'Color', 'Airbags', 'With_Turbo'])
                input_data_cpy = input_data.copy()
                processed_data = preprocess_data(input_data)
                # Make a prediction
                prediction = model_pred.predict(processed_data)
                st.session_state.estimated_price = round(prediction[0])
                # Display the prediction
                time.sleep(4)
                st.session_state.success_message = f'Estimated Price: ${st.session_state.estimated_price}'
                st.success(f'Estimated Price: ${st.session_state.estimated_price}')

                st.session_state.input_data_cpy = input_data_cpy
                st.session_state.prediction_made = True
                st.experimental_rerun()
        if report_button and st.session_state.prediction_made:
            st.session_state.report_page = True
            st.experimental_rerun()
