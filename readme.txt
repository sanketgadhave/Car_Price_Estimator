Introduction:
This Streamlit application is a car price estimator that uses a Gradient Boosting model to predict car prices based on various features like mileage, year of production, manufacturer, and more.

Features:
-Interactive UI to input car features.
-Data preprocessing and transformation.
-Real-time car price estimation.
-Visualizations for price estimation over different production years and categories.

Step 1: Installation:
To run this application locally, you need to install the necessary Python libraries. The required libraries are listed in the requirements.txt file. To install these libraries, run the following command:

pip install -r requirements.txt


Step 2: Running the Application:
After installing the dependencies, you can start the application by running the following command in your terminal:

streamlit run app.py

This will start the Streamlit server and the application should be accessible through your web browser at the address indicated in your terminal (usually http://localhost:8501).

Usage
To use the app:

1. Navigate to the provided URL in your web browser.
2. Enter the details about the car in the input fields.
3. Click 'Get Estimate' to see the predicted price.
4. Use the 'Generate Report' button for a detailed analysis.