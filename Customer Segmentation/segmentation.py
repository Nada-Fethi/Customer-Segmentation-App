import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Models
# -----------------------------
Kmeans = joblib.load("Kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("🧠 Customer Segmentation Predictor")
st.write(
    "Enter the customer's details below to predict the segment they belong to. "
    "This app uses a K-Means model trained on historical customer data."
)

# -----------------------------
# Sidebar: Customer Inputs
# -----------------------------
st.sidebar.header("Customer Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
income = st.sidebar.number_input("Annual Income ($)", min_value=0, max_value=200000, value=50000)
total_spending = st.sidebar.number_input("Total Spending (sum of purchases)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.sidebar.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.sidebar.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.sidebar.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
recency = st.sidebar.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

input_scaled = scaler.transform(input_data)

# -----------------------------
# Define Cluster Profiles
# -----------------------------
cluster_profiles = {
    0: "High-income, loyal customers 🏆",
    1: "Low-income, infrequent spenders 💰",
    2: "Affluent, selective shoppers 💎",
    3: "Older, cautious spenders 🧓",
    4: "Middle-income, occasional buyers 🛒",
    5: "Young, active spenders ⚡"
}

# -----------------------------
# Predict Cluster
# -----------------------------
if st.sidebar.button("Predict Segment"):
    cluster = Kmeans.predict(input_scaled)[0]
    st.success(f"✅ Predicted Segment: Cluster {cluster}")
    st.info(f"Segment Profile: {cluster_profiles.get(cluster, 'Profile not defined')}")

# -----------------------------
# Cluster Summary Dashboard
# -----------------------------
st.header("Cluster Overview Dashboard")

# Simulated cluster summary (replace with real data)
cluster_summary = pd.DataFrame({
    "Cluster": [0, 1, 2, 3, 4, 5],
    "Avg_Age": [60, 48, 62, 69, 50, 51],
    "Avg_Income": [59750, 32078, 74069, 44014, 34435, 78108],
    "Avg_Spending": [900, 104, 1235, 178, 135, 1263]
})

# Display Cluster Cards
st.subheader("Cluster Profiles")
cols = st.columns(3)
for i, col in enumerate(cols):
    for idx in range(i, len(cluster_summary), 3):
        col.metric(
            label=f"Cluster {cluster_summary.loc[idx,'Cluster']}",
            value=f"${int(cluster_summary.loc[idx,'Avg_Income'])} Avg Income",
            delta=f"{int(cluster_summary.loc[idx,'Avg_Spending'])} Avg Spending"
        )
        st.write(cluster_profiles.get(cluster_summary.loc[idx,'Cluster'], ""))

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("Cluster Average Spending")
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Cluster", y="Avg_Spending", data=cluster_summary, palette="Set2", ax=ax)
ax.set_title("Average Spending by Cluster")
st.pyplot(fig)

st.subheader("Cluster Average Income")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(x="Cluster", y="Avg_Income", data=cluster_summary, palette="Set1", ax=ax2)
ax2.set_title("Average Income by Cluster")
st.pyplot(fig2)
