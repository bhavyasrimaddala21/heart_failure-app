import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("❤️ Heart Failure Prediction App")

# Load dataset (MAKE SURE CSV IS IN GITHUB REPO)
@st.cache_data
def load_data():
    return pd.read_csv("heartfailure.csv")

data = load_data()

st.subheader("Dataset Preview")
st.write(data.head())

# Features and target
X = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

# Train models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr_model = LogisticRegression(max_iter=1000)
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()

lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Sidebar input (USER INPUT LIKE YOUR REFERENCE APP)
st.sidebar.header("Enter Patient Details")

def user_input():
    age = st.sidebar.slider("Age", 20, 100, 50)
    anaemia = st.sidebar.selectbox("Anaemia", [0, 1])
    creatinine_phosphokinase = st.sidebar.number_input("CPK", 0, 8000, 200)
    diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
    ejection_fraction = st.sidebar.slider("Ejection Fraction", 10, 80, 30)
    high_blood_pressure = st.sidebar.selectbox("High BP", [0, 1])
    platelets = st.sidebar.number_input("Platelets", 0, 850000, 250000)
    serum_creatinine = st.sidebar.number_input("Serum Creatinine", 0.0, 10.0, 1.0)
    serum_sodium = st.sidebar.number_input("Serum Sodium", 100, 150, 135)
    sex = st.sidebar.selectbox("Sex (0=Female,1=Male)", [0, 1])
    smoking = st.sidebar.selectbox("Smoking", [0, 1])
    time = st.sidebar.slider("Follow-up Days", 0, 300, 100)

    data = {
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": creatinine_phosphokinase,
        "diabetes": diabetes,
        "ejection_fraction": ejection_fraction,
        "high_blood_pressure": high_blood_pressure,
        "platelets": platelets,
        "serum_creatinine": serum_creatinine,
        "serum_sodium": serum_sodium,
        "sex": sex,
        "smoking": smoking,
        "time": time
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input()

st.subheader("User Input")
st.write(input_df)

# Model selection
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "Random Forest"]
)

# Prediction
if model_choice == "Logistic Regression":
    prediction = lr_model.predict(input_df)
elif model_choice == "Decision Tree":
    prediction = dt_model.predict(input_df)
else:
    prediction = rf_model.predict(input_df)

# Output
st.subheader("Prediction Result")

if prediction[0] == 1:
    st.error("⚠️ High Risk of Heart Failure")
else:
    st.success("✅ Low Risk of Heart Failure")
