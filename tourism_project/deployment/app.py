import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Agilandeswari/Tourism_Package_Prediction", filename="tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism package prediction App")
st.write("""
This application predicts the likelihood of a choosing the package the tourism company offers.
Please enter the configuration data below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited ", "M"])
ProdTaken = st.selectbox("ProdTaken", ["1","0"])
CityTier = st.selectbox("CityTier", ["3", "1", "2"])
Occupation = st.selectbox("Occupation", ["Small Business", "Salaried", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=2, max_value=80, value=30, step=1)
NumberOfPersonVisiting = st.number_input("Number Of PersonVisiting", min_value=1, max_value=5, value=2, step=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=1.0, max_value=4.0, value=2.0, step=1.0)
ProductPitched = st.selectbox("ProductPitched", ["Super Deluxe", "Basic", "Standard", "Deluxe","King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["1.0", "2.0", "3.0","4.0","5.0"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced","Unmarried"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=2.0, max_value=10.0, value=4.0, step=1.0)
Passport = st.selectbox("Passport", ["1", "0"])
PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", ["1", "2","3", "4","5"])
OwnCar = st.selectbox("OwnCar", ["1", "0"])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0.0, max_value=4.0, value=2.0, step=1.0)
Designation = st.selectbox("Designation", ["AVP", "Executive", "Senior Manager","Manager","VP"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=200.0, max_value=90000.0, value=24241.0)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5.0, max_value=50.0, value=10.0) 

# Assemble input into DataFrame
input_data = pd.DataFrame(
    [
    {
    'TypeofContact': TypeofContact,
    'ProdTaken': ProdTaken,
    'CityTier': CityTier,
    'Gender': Gender,
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips':NumberOfTrips,
    'Passport':Passport,
    'PitchSatisfactionScore':PitchSatisfactionScore,
    'OwnCar':OwnCar,
    'NumberOfChildrenVisiting':NumberOfChildrenVisiting,
    'Designation':Designation,
    'Occupation': Occupation
    'MonthlyIncome':MonthlyIncome,
    'DurationOfPitch': DurationOfPitch

}])


if st.button("Tourism package prediction"):
    prediction = model.predict(input_data)[0]
    result = "Tourism package" if prediction == 1 else "Pacakge will not be taken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
