import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Streamlit UI
st.title("Employee Salary Prediction App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head())

    # Feature and target selection
    st.write("### Select Features (Independent variables):")
    selected_features = st.multiselect("Choose columns for features (X):", data.columns.tolist())

    st.write("### Select Target (Dependent variable):")
    selected_target = st.selectbox("Choose column for target (y):", data.columns.tolist())

    if selected_features and selected_target:
        X = data[selected_features]
        y = data[selected_target]

        # Encode target labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predict and show accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Model Accuracy: {accuracy:.2f}")
