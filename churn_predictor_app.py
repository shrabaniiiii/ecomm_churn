
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the trained model and preprocessors
@st.cache_resource
def load_resources():
    model = joblib.load('random_forest_churn_model.sav')
    scaler = joblib.load('scaler.sav')
    label_encoders = joblib.load('label_encoders.joblib')
    return model, scaler, label_encoders

model, scaler, label_encoders = load_resources()

# Streamlit App Title
st.title('Customer Churn Prediction App')
st.write('Enter customer details to predict if they will churn.')

# Input fields for features
st.sidebar.header('Customer Features')

# Reconstruct the feature list from X used during training (excluding 'churn_target' and 'subscription_status')
# These columns were present in the final X_train, before scaling
# Some columns like order_id, customer_id, product_id, signup_date, last_purchase_date, order_date are generally not used as direct features or handled differently.
# Based on the original df.info() and df.head() and typical feature engineering for churn, let's select relevant inputs.

# Numerical features from the original dataset
age = st.sidebar.slider('Age', 18, 70, 30)
cancellations_count = st.sidebar.slider('Cancellations Count', 0, 5, 1)
unit_price = st.sidebar.number_input('Unit Price', value=100.0, step=1.0)
quantity = st.sidebar.slider('Quantity', 1, 10, 5)
purchase_frequency = st.sidebar.slider('Purchase Frequency (days)', 1, 50, 25)

# Categorical features - using unique values from the original DataFrame for options
# For simplicity, we are getting unique values directly from the original_df or using some reasonable defaults.
# In a real scenario, you'd save these original unique values with your encoders.
# For 'country', 'preferred_category', 'product_name', 'category', 'gender' we have their LabelEncoders.

country_options = list(label_encoders['country'].classes_)
country = st.sidebar.selectbox('Country', options=country_options, index=0)

preferred_category_options = list(label_encoders['preferred_category'].classes_)
preferred_category = st.sidebar.selectbox('Preferred Category', options=preferred_category_options, index=0)

product_name_options = list(label_encoders['product_name'].classes_)
product_name = st.sidebar.selectbox('Product Name', options=product_name_options, index=0)

category_options = list(label_encoders['category'].classes_)
category = st.sidebar.selectbox('Category', options=category_options, index=0)

gender_options = list(label_encoders['gender'].classes_)
gender = st.sidebar.selectbox('Gender', options=gender_options, index=0)

# Placeholder values for features that were label encoded but not in the original X (e.g., customer_id, order_id, product_id)
# These were also encoded in the original data, but usually dropped if they are just identifiers or not directly predictive.
# If they were kept in X during training, we need to handle them. Based on af5b9e92, they were dropped.
# However, if 'order_id', 'customer_id', 'product_id' were in X, they would have been label encoded. Let's assume they were not direct features.
# If they were to be included, you would need to either: 
# 1. Provide input for them (e.g., enter new customer_id) and encode.
# 2. Assign a default value (e.g., 0) after encoding. This is risky for new data.

# For the sake of this app, I'm assuming 'order_id', 'customer_id', 'product_id' were indeed dropped from X.
# The remaining 3 object columns 'signup_date', 'last_purchase_date', 'order_date' were also label encoded in a6b9cff3.
# We need to provide inputs for these if they are features in X.

# Re-evaluating the `X` features from the notebook (af5b9e92, after label encoding, before splitting/scaling)
# `X = df.drop(columns=['churn_target', 'subscription_status'])`
# The `df` at that point had all original columns (except 'churn_target', 'subscription_status'), but already label encoded.
# So 'order_id', 'customer_id', 'product_id', 'signup_date', 'last_purchase_date', 'order_date' were *all* part of X.
# This means we need to handle these in the Streamlit app. For simplicity, we'll ask for numerical inputs as they were already encoded.

# Input for ID-like features (which were label encoded earlier)
order_id_input = st.sidebar.number_input('Order ID (encoded)', value=1000, min_value=0)
customer_id_input = st.sidebar.number_input('Customer ID (encoded)', value=1000, min_value=0)
product_id_input = st.sidebar.number_input('Product ID (encoded)', value=1000, min_value=0)

# Input for date features (which were label encoded earlier)
signup_date_input = st.sidebar.number_input('Signup Date (encoded)', value=500, min_value=0)
last_purchase_date_input = st.sidebar.number_input('Last Purchase Date (encoded)', value=500, min_value=0)
order_date_input = st.sidebar.number_input('Order Date (encoded)', value=500, min_value=0)


if st.sidebar.button('Predict Churn'):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'order_id': order_id_input,
        'customer_id': customer_id_input,
        'age': age,
        'product_id': product_id_input,
        'country': country,
        'signup_date': signup_date_input,
        'last_purchase_date': last_purchase_date_input,
        'cancellations_count': cancellations_count,
        'order_date': order_date_input,
        'unit_price': unit_price,
        'quantity': quantity,
        'purchase_frequency': purchase_frequency,
        'preferred_category': preferred_category,
        'product_name': product_name,
        'category': category,
        'gender': gender
    }])

    # Apply Label Encoding for categorical features in input_data
    for col, le in label_encoders.items():
        # Handle unseen labels by mapping them to a default (e.g., 0) or raising an error
        # For simplicity, if a label is unseen, we'll use a placeholder. 
        # A more robust app would handle this carefully.
        if col in input_data.columns:
            try:
                input_data[col] = le.transform(input_data[col])
            except ValueError: # If an unseen label is encountered
                st.warning(f"Unseen label for {col}. Defaulting to 0.")
                input_data[col] = 0 # Or use a more appropriate default

    # Scale numerical features
    # The order of columns in input_data must match the order X was trained on.
    # Let's define the exact columns used in X in the notebook.
    # X = df.drop(columns=['churn_target', 'subscription_status'])
    # The columns were: 'order_id', 'customer_id', 'age', 'product_id', 'country', 'signup_date',
    #                   'last_purchase_date', 'cancellations_count', 'order_date', 'unit_price',
    #                   'quantity', 'purchase_frequency', 'preferred_category', 'product_name',
    #                   'category', 'gender'

    # Ensure column order matches training data's X
    feature_columns = ['order_id', 'customer_id', 'age', 'product_id', 'country', 'signup_date',
                       'last_purchase_date', 'cancellations_count', 'order_date', 'unit_price',
                       'quantity', 'purchase_frequency', 'preferred_category', 'product_name',
                       'category', 'gender']
    
    # Apply scaler. Ensure data type is float before scaling.
    scaled_input = scaler.transform(input_data[feature_columns].values.astype(float))

    # Make prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f"The customer is predicted to CHURN with probability {prediction_proba[0][1]:.2f}")
    else:
        st.success(f"The customer is predicted to NOT CHURN with probability {prediction_proba[0][0]:.2f}")

    st.write("Note: Probability for Churn is the second value in predict_proba: ", prediction_proba[0][1])
