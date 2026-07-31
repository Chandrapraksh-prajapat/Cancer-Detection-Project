
import streamlit as st
import requests

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Cancer Detection App", layout="wide")

st.title("🩺 Breast Cancer Detection")
st.write(
    "Enter the tumor measurements below and click **Predict** "
    "to get the model's prediction from the FastAPI backend."
)

# ---------------------------------------------------------
# Feature groups
# ---------------------------------------------------------
mean_features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
]

se_features = [
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
]

worst_features = [
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

all_features = mean_features + se_features + worst_features

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Settings")

api_url_input = st.sidebar.text_input(
    "Backend API URL",
    value=API_URL
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Tip: Run the backend using:\n\n"
    "```bash\n"
    "python -m uvicorn main:app --reload\n"
    "```"
)

# ---------------------------------------------------------
# Input Form
# ---------------------------------------------------------
input_values = {}

with st.form("prediction_form"):

    tab_mean, tab_se, tab_worst = st.tabs(
        ["Mean", "Standard Error", "Worst"]
    )

    def render_inputs(tab, features):
        with tab:
            cols = st.columns(2)

            for i, feature in enumerate(features):

                label = feature.replace("_", " ").title()

                with cols[i % 2]:

                    input_values[feature] = st.number_input(
                        label,
                        value=0.0,
                        format="%.5f",
                        key=feature
                    )

    render_inputs(tab_mean, mean_features)
    render_inputs(tab_se, se_features)
    render_inputs(tab_worst, worst_features)

    submitted = st.form_submit_button("🔍 Predict")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    payload = {
        feature: input_values[feature]
        for feature in all_features
    }

    with st.spinner("Contacting Model API..."):

        try:

            response = requests.post(
                api_url_input,
                json=payload,
                timeout=15
            )

            response.raise_for_status()

            result = response.json()

            prediction = result.get("prediction")

            st.subheader("Prediction Result")

            if isinstance(prediction, list):
                pred = prediction[0]
            else:
                pred = prediction

            if pred == "M":
                st.error("🔴 Malignant (Cancer Detected)")
            elif pred == "B":
                st.success("🟢 Benign (No Cancer Detected)")
            else:
                st.success(f"Prediction : {pred}")

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to FastAPI.\n\n"
                "Please start the backend first:\n\n"
                "python -m uvicorn main:app --reload"
            )

        except requests.exceptions.HTTPError as e:

            st.error(f"API Error:\n\n{e}")

        except Exception as e:

            st.error(f"Unexpected Error:\n\n{e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "This application is for educational purposes only and "
    "should not be used for professional medical diagnosis."
)