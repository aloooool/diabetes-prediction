import streamlit as st
import random
import time

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Diabetes Predictor Pro",
    page_icon="🩺",
    layout="centered"
)


# 3. Modular Prediction Function
def predict_diabetes(features: dict) -> int:
    """Simulates a machine learning inference step."""
    #all model logic import from other file
    # When ready, replace this with: return model.predict(pd.DataFrame([features]))[0]
    return random.choice([0, 1])

# 4. Main Application Application
def main():
    st.title("🩺 Diabetes Prediction Portal")
    st.markdown("Enter the patient's diagnostic metrics below to generate a risk assessment.")
    
    
    # 5. Form to batch inputs and prevent constant page reloads
    with st.form("prediction_form"):
        st.subheader("Patient Vitals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, step=1, help="Number of times pregnant")
            glucose = st.number_input("Glucose", min_value=0, help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)")
            bp = st.number_input("Blood Pressure", min_value=0, help="Diastolic blood pressure (mm Hg)")
            skin_thickness = st.number_input("Skin Thickness", min_value=0, help="Triceps skin fold thickness (mm)")
            
        with col2:
            insulin = st.number_input("Insulin", min_value=0, help="2-Hour serum insulin (mu U/ml)")
            bmi = st.number_input("BMI", min_value=0.0, format="%.1f", help="Body mass index (weight in kg/(height in m)^2)")
            dpf = st.number_input("Diabetes Pedigree", min_value=0.000, format="%.3f", help="Diabetes pedigree function scoring genetic risk")
            age = st.number_input("Age", min_value=0, step=1, help="Patient age in years")
            
        # Submit button for the form
        submitted = st.form_submit_button("Run Prediction", type="primary")

    if submitted:
        # 6. Package features as a dictionary for clean data handling
        features = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp, 
            "SkinThickness": skin_thickness, "Insulin": insulin, 
            "BMI": bmi, "DiabetesPedigreeFunction": dpf, "Age": age
        }
        
        
        with st.spinner("Analyzing patient metrics..."):
            time.sleep(0.5) 
            result = predict_diabetes(features)
            
            
        if result == 1:
            st.error("### 🚨 High Risk\nThe model predicts the patient is **positive** for diabetes.")
            st.info("Recommendation: Consult a healthcare professional for a formal clinical assessment.")
        else:
            st.success("### ✅ Low Risk\nThe model predicts the patient is **negative** for diabetes.")

if __name__ == "__main__":
    main()