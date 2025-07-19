import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Streamlit UI
st.title("Employee Salary Prediction App")

# Automatically load the CSV that is already in the repo
df = pd.read_csv("employee_data.csv")
st.success("✅ Dataset loaded successfully from repository.")

    st.write("### Data Preview", data.head())

    # Feature and target selection
    st.write("### Select Features (Independent variables):")
    selected_features = st.multiselect("Choose columns for features (X):", data.columns.tolist())

    st.write("### Select Target (Dependent variable):")
    selected_target = st.selectbox("Choose column for target (y):", data.columns.tolist())

    if selected_features and selected_target:
        X = data[selected_features]
        y = data[selected_target]
        if st.button("Train Model"):
            # Step 1: Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Step 2: Create the model
            model = LogisticRegression()

            # Step 3: Train the model
            model.fit(X_train, y_train)

            # Step 4: Make predictions
            y_pred = model.predict(X_test)

            # Step 5: Show accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model Trained Successfully!\n\nAccuracy: {accuracy * 100:.2f}%")

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
