import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk

# Set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🕵️",
    layout="wide"
)

# Load the saved pipeline
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load("fraud_detection_pipeline.pkl")
        st.success("Pipeline loaded successfully!")
        return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None

pipeline = load_pipeline()

# App title and description
st.title("🕵️ Transaction Fraud Detection System")
st.markdown("""
This application developed by Similoluwa Folorunso detects potentially fraudulent transactions using machine learning.
""")

# Navigation
page = st.sidebar.radio("Navigation", ["Real-time Prediction", "Batch Processing"])

if page == "Real-time Prediction":
    st.header("🔍 Single Transaction Analysis")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            merchant = st.text_input("Merchant ID", "M12345")
            category = st.selectbox("Category", ["grocery", "entertainment", "gas_transport", "shopping", "food_dining"])
            amount = st.number_input("Amount ($)", min_value=0.01, value=150.0)
            last = st.number_input("Days Since Last Transaction", min_value=0, value=7)
            gender = int(st.radio("Gender", ("1", "0")))
            
        with col2:
            job = st.selectbox("Occupation", ["Engineer", "Doctor", "Teacher", "Executive", "Analyst","Others"])
            lat = st.number_input("Customer Latitude", value=40.71)
            long = st.number_input("Customer Longitude", value=-74.01)
            merch_lat = st.number_input("Merchant Latitude", value=40.72)
            merch_long = st.number_input("Merchant Longitude", value=-74.02)
            city_pop = st.number_input("City Population", min_value=0, value=1000000)
            hour = st.slider("Transaction Hour", 0, 23, 14)
            month = st.slider("Month", 1, 12, datetime.now().month)
        
        submitted = st.form_submit_button("Check Fraud Risk")
        
    if submitted and pipeline:
        # Create input DataFrame
        input_data = pd.DataFrame([[merchant, category, amount, last, gender, 
                                  lat, long, city_pop, job, merch_lat, 
                                  merch_long, hour, month]],
            columns=['merchant', 'category', 'amt', 'last', 'gender', 
                     'lat', 'long', 'city_pop', 'job', 'merch_lat', 
                     'merch_long', 'hour', 'month'])
        
        # Make prediction
        try:
            proba = pipeline.predict_proba(input_data)[0][1]
            
            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)
            col1.metric("Fraud Probability", f"{proba*100:.2f}%")
            
            risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"
            col2.metric("Risk Level", risk_level)
            
            # Visual gauge
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Risk'], [proba], color='red' if proba > 0.7 else 'orange' if proba > 0.3 else 'green')
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.text(proba/2, 0, f"{proba*100:.1f}%", 
                   ha='center', va='center', color='white', fontsize=20)
            st.pyplot(fig)
            
            # Recommendation
            if proba > 0.7:
                st.error("🚨 High fraud risk detected!")
                st.markdown("""
                **Recommended actions:**
                - Verify customer identity
                - Request additional authentication
                - Review transaction history
                """)
            elif proba > 0.3:
                st.warning("⚠️ Moderate fraud risk detected")
                st.markdown("""
                **Recommended actions:**
                - Send verification code
                - Check recent transactions
                - Monitor account
                """)
            else:
                st.success("✅ Low fraud risk detected")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")

elif page == "Batch Processing":
    st.header("📂 Batch Transaction Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file with transactions", type="csv")
    
    if uploaded_file and pipeline:
        try:
            batch_data = pd.read_csv(uploaded_file)
            
            # Ensure required columns exist
            required_cols = ['merchant', 'category', 'amt', 'last', 'gender',
                           'lat', 'long', 'city_pop', 'job', 'merch_lat',
                           'merch_long', 'hour', 'month']
            
            if all(col in batch_data.columns for col in required_cols):
                # Make predictions
                predictions = pipeline.predict_proba(batch_data[required_cols])[:,1]
                batch_data['fraud_probability'] = predictions
                batch_data['fraud_alert'] = np.where(predictions > 0.7, 'High', 
                                                    np.where(predictions > 0.3, 'Medium', 'Low'))
                
                # Show results
                st.dataframe(batch_data.sort_values('fraud_probability', ascending=False))
                
                # Download results
                csv = batch_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )
                
                # Summary stats
                st.subheader("Batch Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", len(batch_data))
                col2.metric("High Risk Transactions", f"{sum(predictions > 0.7)} ({sum(predictions > 0.7)/len(predictions)*100:.1f}%)")
                col3.metric("Average Fraud Probability", f"{np.mean(predictions)*100:.2f}%")
                
            else:
                st.error(f"Missing required columns. Needed: {required_cols}")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.warning("Pipeline not loaded")

# Footer
st.markdown("---")

st.markdown("Fraud Detection System Developed by Kolaru Gideon Mosimiloluwa")

