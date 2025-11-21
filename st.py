import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="Customer Spend Prediction App", layout="wide")
st.title("🛒 Customer Product Spend Prediction App")

MODEL_FILES = {
    "MntWines": "model_1.pkl",
    "MntFruits": "model_2.pkl",
    "MntMeatProducts": "model_3.pkl",
    "MntFishProducts": "model_4.pkl",
    "MntSweetProducts": "model_5.pkl",
    "MntGoldProds": "model_6.pkl"
}

models = {}

for label, file in MODEL_FILES.items():
    if os.path.exists(file):
        models[label] = joblib.load(file)
    else:
        st.error(f"Model file missing: {file}")

FEATURES = {
    "MntWines": ['NumStorePurchases','NumCatalogPurchases','Income','NumWebPurchases',
                 'Kidhome','AcceptedCmp5','AcceptedCmp4','NumWebVisitsMonth',
                 'AcceptedCmp2','Education_PhD','Education_2n Cycle','Education_Graduation'],

    "MntFruits": ['NumCatalogPurchases','NumStorePurchases','Income','NumWebVisitsMonth',
                  'Teenhome','Education_PhD','Education_Master','AcceptedCmp2'],

    "MntMeatProducts": ['NumCatalogPurchases','Income','NumWebVisitsMonth','NumStorePurchases',
                        'AcceptedCmp5','Teenhome','Education_Graduation','AcceptedCmp2'],

    "MntFishProducts": ['NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','Income',
                        'Teenhome','Education_PhD','Education_Master','AcceptedCmp4'],

    "MntSweetProducts": ['NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','NumWebPurchases',
                         'AcceptedCmp1','Teenhome','Education_PhD','Education_Master'],

    "MntGoldProds": ['NumCatalogPurchases','NumStorePurchases','Kidhome','Education_Graduation']
}

st.header("Enter Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    Year_Birth = st.number_input("Year of Birth", min_value=1900, max_value=2025, value=1980)
    Education = st.selectbox("Education", ["2n Cycle","Basic","Graduation","Master","PhD"])
    Marital_Status = st.selectbox("Marital Status",
                                  ["Single","Married","Together","Divorced","Widow","Alone","Absurd","YOLO"])
    Income = st.number_input("Income", min_value=0, value=50000)
    Kidhome = st.number_input("Kidhome", 0, 25, 0)
    Teenhome = st.number_input("Teenhome", 0, 50, 0)

with col2:
    Recency = st.number_input("Recency", 0, 10000000, 50)
    NumDealsPurchases = st.number_input("NumDealsPurchases", 0, 1000000000, 5)
    NumWebPurchases = st.number_input("NumWebPurchases", 0, 1000000000, 5)
    NumCatalogPurchases = st.number_input("NumCatalogPurchases", 0, 1000000000, 5)
    NumStorePurchases = st.number_input("NumStorePurchases", 0, 1000000000, 5)
    NumWebVisitsMonth = st.number_input("NumWebVisitsMonth", 0, 1000000000, 5)

with col3:
    AcceptedCmp1 = st.selectbox("AcceptedCmp1", [0,1])
    AcceptedCmp2 = st.selectbox("AcceptedCmp2", [0,1])
    AcceptedCmp3 = st.selectbox("AcceptedCmp3", [0,1])
    AcceptedCmp4 = st.selectbox("AcceptedCmp4", [0,1])
    AcceptedCmp5 = st.selectbox("AcceptedCmp5", [0,1])
    Complain = st.selectbox("Complain", [0,1])
    Response = st.selectbox("Response", [0,1])

input_df = pd.DataFrame([{
    "Year_Birth": Year_Birth,
    "Education": Education,
    "Marital_Status": Marital_Status,
    "Income": Income,
    "Kidhome": Kidhome,
    "Teenhome": Teenhome,
    "Recency": Recency,
    "NumDealsPurchases": NumDealsPurchases,
    "NumWebPurchases": NumWebPurchases,
    "NumCatalogPurchases": NumCatalogPurchases,
    "NumStorePurchases": NumStorePurchases,
    "NumWebVisitsMonth": NumWebVisitsMonth,
    "AcceptedCmp1": AcceptedCmp1,
    "AcceptedCmp2": AcceptedCmp2,
    "AcceptedCmp3": AcceptedCmp3,
    "AcceptedCmp4": AcceptedCmp4,
    "AcceptedCmp5": AcceptedCmp5,
    "Complain": Complain,
    "Response": Response
}])
input_df_2=input_df.copy()

input_df = pd.get_dummies(input_df, columns=['Education','Marital_Status'], drop_first=True)

all_dummy_cols = [
    'Education_PhD','Education_Master','Education_Graduation','Education_2n Cycle',
    'Education_Basic','Education_High School','Education_Elementary','Education_Unknown',
    'Marital_Status_Single','Marital_Status_Married','Marital_Status_Together',
    'Marital_Status_Divorced','Marital_Status_Widow'
]

for col in all_dummy_cols:
    if col not in input_df.columns:
        input_df[col] = 0

cluster_scaler = joblib.load("cluster_scaler.pkl")
cluster_model = joblib.load("cluster_model.pkl")

if st.button("Predict Customer Spending"):
    st.subheader("**Predicted Spend Amounts**")

    results = {}

    for label, model in models.items():
        req_features = FEATURES[label]

        for f in req_features:
            if f not in input_df.columns:
                input_df[f] = 0

        X_pred = input_df[req_features]

        pred_value = float(model.predict(X_pred)[0])
        if pred_value < 0:
            pred_value = 0
        results[label] = round(pred_value, 2)

    st.success("Prediction completed successfully!")

    st.write("### Estimated Spending (Customer will spend):")
    st.json(results)
    total_spending = sum(results.values())
    st.subheader("Total Estimated Spending")
    st.success(f"Total Spending: {round(total_spending, 2)}")

    st.subheader("🧩 Customer Segment Prediction and it's Meaning")
    with open("education_map.json") as f:
        education_map = json.load(f)

    with open("marital_map.json") as f:
        marital_map = json.load(f)

    input_df_2['Education_encoded'] = input_df_2['Education'].map(education_map)
    input_df_2['Marital_Status_encoded'] = input_df_2['Marital_Status'].map(marital_map)

    input_df_2['Total_Purchases'] = (
        input_df_2['NumCatalogPurchases'] +input_df_2['NumStorePurchases'] +input_df_2['NumWebPurchases'] 
        + input_df_2['NumDealsPurchases'])
    
    cluster_features = [
        'Income','Education_encoded','Marital_Status_encoded','Kidhome',
        'Total_Purchases','Response','NumCatalogPurchases',
        'NumStorePurchases','AcceptedCmp5'
    ]

    cluster_input = input_df_2[cluster_features]
    cluster_scaled = cluster_scaler.transform(cluster_input)
    cluster_label = int(cluster_model.predict(cluster_scaled)[0])

    st.success(f" The customer belongs to **Cluster {cluster_label}**")

    cluster_info = {
        0: {
            "Segment Name": "High Income Frequent Shoppers",
            "Key Traits": "High spend, low response, store-oriented",
            "Value to Business": "High value but difficult to influence with promos"
        },
        1: {
            "Segment Name": "Value Seekers with Kids",
            "Key Traits": "Low income, many kids, low spend, no response",
            "Value to Business": "Lowest value; very price-sensitive"
        },
        2: {
            "Segment Name": "Catalog-Oriented Loyal Responders",
            "Key Traits": "Mid-income, multichannel, highly responsive",
            "Value to Business": "Strong value; ideal for targeted campaigns"
        },
        3: {
            "Segment Name": "Affluent Multichannel Big Spenders",
            "Key Traits": "Highest income, high spend, respond to campaigns",
            "Value to Business": "Highest value; perfect for premium targeting"
        }
    }

    seg = cluster_info.get(cluster_label, None)

    if seg:
        st.success(f"**Cluster {cluster_label} — {seg['Segment Name']}**")
        
        st.write(f"**Key Traits:** {seg['Key Traits']}")
        st.write(f"**Value to Business:** {seg['Value to Business']}")
    else:
        st.warning("⚠ No description available for this cluster.")


