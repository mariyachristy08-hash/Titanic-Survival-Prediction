
import streamlit as st
import pandas as pd
import joblib

# Feature order must match Project 04
FEATURES = ["pclass", "sex", "age", "fare"]

# Load the trained model only once
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


# Website heading
st.title("Titanic Survival Prediction")

st.write(
    "Enter passenger details below to estimate "
    "the chance of survival."
)

st.caption(
    "Educational project using a Decision Tree "
    "trained on the historical Titanic dataset."
)

# Passenger inputs
pclass = st.selectbox(
    "Passenger Ticket Class",
    [1, 2, 3]
)

sex = st.radio(
    "Passenger Sex",
    ["female", "male"]
)

age = st.slider(
    "Passenger Age",
    min_value=1,
    max_value=80,
    value=30
)

fare = st.slider(
    "Ticket Fare",
    min_value=0,
    max_value=500,
    value=30
)

# Prediction button
if st.button("Predict Survival"):

    try:
        model = load_model()

        # Prepare input in the same feature order
        passenger = pd.DataFrame(
            [[
                pclass,
                1 if sex == "female" else 0,
                age,
                fare
            ]],
            columns=FEATURES
        )

        # Predict survival probability
        chance = model.predict_proba(passenger)[0][1]

        # Display prediction
        st.metric(
            "Estimated Chance of Survival",
            f"{chance:.0%}"
        )

        if chance >= 0.5:
            st.success("The model predicts: Survived")
        else:
            st.warning("The model predicts: Did not survive")

        st.caption(
            "This is an educational prediction based on "
            "historical data, not a real-world assessment."
        )

    except FileNotFoundError:
        st.error(
            "Model file is missing. Please add model.joblib "
            "to the project folder."
        )

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.caption(str(error))
