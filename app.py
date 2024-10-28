import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import MinMaxScalerModel, VectorAssembler
from pyspark.ml import PipelineModel

# Create Spark session
spark = SparkSession.builder \
    .appName("Credit Risk Classification") \
    .getOrCreate()

# Load the trained Random Forest model and scaler
rf_model = RandomForestClassificationModel.load("models/random_forest_credit_risk")
scaler_model = MinMaxScalerModel.load("models/minmax_scaler")

# Load the preprocessing pipeline
pipeline_model = PipelineModel.load("models/preprocessing_pipeline")

# Input form for user data
st.title("Credit Risk Prediction")
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "DEBT_CONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

# Numerical inputs
person_age = st.number_input("Age", min_value=0)
person_income = st.number_input("Income", min_value=0.0)
person_emp_length = st.number_input("Employment Length (in years)", min_value=0)
loan_amnt = st.number_input("Loan Amount", min_value=0.0)
loan_int_rate = st.number_input("Interest Rate", min_value=0.0)
cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=0)

# Prediction button
if st.button("Predict"):
    # Create a DataFrame for input data
    input_data = spark.createDataFrame([(person_home_ownership, loan_intent, loan_grade,
                                          person_age, person_income, person_emp_length,
                                          loan_amnt, loan_int_rate, cb_person_cred_hist_length)],
                                        ["person_home_ownership", "loan_intent", "loan_grade",
                                         "person_age", "person_income", "person_emp_length",
                                         "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"])
    
    # Apply the preprocessing pipeline to the input data
    input_data = pipeline_model.transform(input_data)

    # Assemble features
    assembler = VectorAssembler(inputCols=['person_age', 'person_income', 'person_emp_length',
                                           'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length',
                                           'person_home_ownership_encoded', 'loan_intent_encoded', 'loan_grade_encoded'],
                                outputCol="features_unscaled")
    
    input_data = assembler.transform(input_data)

    # Scale the features
    input_data = scaler_model.transform(input_data)

    # Make predictions
    predictions = rf_model.transform(input_data)
    
    # Show the prediction result
    result = predictions.select("prediction").first()[0]
    st.success(f"Predicted Loan Status: {'Approved' if result == 1 else 'Denied'}")
