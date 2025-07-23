# ======================
# 📌 1. Imports
# ======================
import streamlit as st
import pandas as pd
import joblib

# ======================
# 📌 2. Load files
# ======================
data = pd.read_csv("adult_encoded.csv")
model = joblib.load("random_forest_model.pkl")
encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")  # ✅ NEW: exact feature order!

# ======================
# 📌 3. App title
# ======================
st.title("🧑‍💼 Employee Salary Prediction App")
st.markdown("Predict whether a person's salary is `<=50K` or `>50K` based on various features.")

if st.checkbox("Show Raw Data"):
    st.dataframe(data.head())

# ======================
# 📌 4. Original text categories
# ======================
workclass_options = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
]

education_options = [
    "Bachelors", "HS-grad", "11th", "Masters", "9th",
    "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
    "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
    "Preschool", "12th"
]

marital_status_options = [
    "Married-civ-spouse", "Divorced", "Never-married",
    "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
]

occupation_options = [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
]

relationship_options = [
    "Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"
]

race_options = [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
    "Other", "Black"
]

gender_options = ["Female", "Male"]

native_country_options = [
    "United-States", "Cambodia", "England", "Puerto-Rico",
    "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India",
    "Japan", "Greece", "South", "China", "Cuba", "Iran",
    "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France",
    "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
    "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua",
    "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
    "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
]

# ======================
# 📌 5. Input widgets
# ======================
age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", workclass_options)
education = st.selectbox("Education", education_options)
marital_status = st.selectbox("Marital Status", marital_status_options)
occupation = st.selectbox("Occupation", occupation_options)
relationship = st.selectbox("Relationship", relationship_options)
race = st.selectbox("Race", race_options)
gender = st.selectbox("Gender", gender_options)
native_country = st.selectbox("Native Country", native_country_options)
hours_per_week = st.slider("Hours per week", 1, 100, 40)

capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
educational_num = st.number_input("Educational Number", min_value=1, max_value=20, value=10)
fnlwgt = st.number_input("Fnlwgt", min_value=0, value=100000)

# ======================
# 📌 6. Format input
# ======================
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'education': [education],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country],
    'hours-per-week': [hours_per_week],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'educational-num': [educational_num],
    'fnlwgt': [fnlwgt]
})

# ======================
# 📌 7. Encode categoricals
# ======================
for col in input_data.columns:
    if col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

# ======================
# ✅✅✅ 8. Use exact feature order saved during training
# ======================
input_data = input_data[feature_order]

# ======================
# 📌 9. Scale input
# ======================
input_scaled = scaler.transform(input_data)

# ======================
# 📌 10. Predict
# ======================
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"🎯 Predicted Salary: {result}")
