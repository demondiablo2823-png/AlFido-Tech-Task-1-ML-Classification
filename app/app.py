import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="💰",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "raw" / "insurance.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"

FALLBACK_MODEL_FILES = {
    "Linear Regression": "linear_regression_pipeline.pkl",
    "Ridge": "ridge_pipeline.pkl",
    "Lasso": "lasso_pipeline.pkl",
    "ElasticNet": "elasticnet_pipeline.pkl",
}


@st.cache_resource
def load_model(filename):
    return joblib.load(MODELS_DIR / filename)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def get_available_models():
    metrics_data = load_metrics()

    if metrics_data and "models" in metrics_data:
        models = {
            name: info for name, info in metrics_data["models"].items()
            if (MODELS_DIR / info["file"]).exists()
        }
        return models, metrics_data.get("best_model")

    models = {
        name: {"file": filename, "r2": "N/A", "rmse": "N/A", "mae": "N/A"}
        for name, filename in FALLBACK_MODEL_FILES.items()
        if (MODELS_DIR / filename).exists()
    }
    return models, None


def make_prediction(model, age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region
    }])
    return model.predict(input_df)[0]


try:
    dataset = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

available_models, best_model_name = get_available_models()

if not available_models:
    st.error("No trained model files found in the models directory.")
    st.stop()

st.sidebar.title("⚙️ Prediction Settings")

st.sidebar.subheader("Model")
model_options = list(available_models.keys())
default_index = model_options.index(best_model_name) if best_model_name in model_options else 0

selected_model_name = st.sidebar.selectbox(
    "Model", model_options, index=default_index,
    label_visibility="collapsed",
    help="Pick which trained regression model to use for the prediction below."
)

if selected_model_name == best_model_name:
    st.sidebar.caption(f"⭐ Best model - R² = {available_models[selected_model_name]['r2']}")

try:
    model = load_model(available_models[selected_model_name]["file"])
except Exception as e:
    st.error(f"Error loading model '{selected_model_name}': {e}")
    st.stop()

st.sidebar.subheader("Personal Information")
col1, col2 = st.sidebar.columns(2)
age = col1.number_input("Age", min_value=18, max_value=100, value=25)
sex = col2.selectbox("Gender", ["male", "female"])

st.sidebar.subheader("Health Information")
col1, col2 = st.sidebar.columns(2)
bmi = col1.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
smoker = col2.selectbox("Smoker", ["yes", "no"])
if bmi < 15 or bmi > 50:
    st.sidebar.caption("⚠️ BMI outside typical range.")

st.sidebar.subheader("Other Information")
col1, col2 = st.sidebar.columns(2)
children = col1.number_input("Children", min_value=0, max_value=10, value=0)
region = col2.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

predict_clicked = st.sidebar.button("Predict Premium", use_container_width=True)

st.title("💰 Insurance Premium Prediction")
st.write("Predict medical insurance charges using a trained regression model of your choice.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Information")
    st.info(f"""
**Model:** {selected_model_name}

**Dataset:** Medical Insurance Cost Dataset

**Target Variable:** Insurance Charges
""")

with col2:
    st.subheader("Prediction")

    if predict_clicked:
        try:
            st.session_state["prediction"] = make_prediction(
                model, age, sex, bmi, children, smoker, region
            )
            st.session_state["prediction_model"] = selected_model_name
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if "prediction" in st.session_state:
        st.metric(
            label=f"Estimated Insurance Charges ({st.session_state['prediction_model']})",
            value=f"${st.session_state['prediction']:,.2f}"
        )
        if st.session_state["prediction_model"] != selected_model_name:
            st.caption("⚠️ Model changed since last prediction. Click **Predict Premium** to update.")
    else:
        st.warning("Enter the details in the sidebar and click **Predict Premium**.")

st.divider()

st.subheader("Input Summary")
st.dataframe(
    pd.DataFrame({
        "Feature": ["Age", "Gender", "BMI", "Children", "Smoker", "Region"],
        "Value": [age, sex, bmi, children, smoker, region]
    }),
    use_container_width=True
)
st.divider()

tab1, tab2, tab3 = st.tabs(["📄 Dataset", "📊 Visualizations", "ℹ️ About Model"])

with tab1:
    view = st.selectbox(
        "Select View",
        ["First 10 Rows", "Dataset Information", "Statistical Summary", "Full Dataset"]
    )

    if view == "First 10 Rows":
        st.dataframe(dataset.head(10), use_container_width=True)
    elif view == "Dataset Information":
        info = pd.DataFrame({
            "Column": dataset.columns,
            "Data Type": dataset.dtypes.astype(str),
            "Missing Values": dataset.isnull().sum().values
        })
        st.dataframe(info, use_container_width=True)
    elif view == "Statistical Summary":
        st.dataframe(dataset.describe(), use_container_width=True)
    else:
        st.dataframe(dataset, use_container_width=True)

with tab2:
    plot = st.selectbox(
        "Choose Visualization",
        ["Age vs Charges", "BMI vs Charges", "Smoker vs Charges", "Region Distribution", "Charges Distribution"]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    if plot == "Age vs Charges":
        ax.scatter(dataset["age"], dataset["charges"], alpha=0.6)
        ax.set(xlabel="Age", ylabel="Charges", title="Age vs Charges")

    elif plot == "BMI vs Charges":
        ax.scatter(dataset["bmi"], dataset["charges"], alpha=0.6)
        ax.set(xlabel="BMI", ylabel="Charges", title="BMI vs Charges")

    elif plot == "Smoker vs Charges":
        dataset.boxplot(column="charges", by="smoker", ax=ax)
        fig.suptitle("")
        ax.set(xlabel="Smoker", ylabel="Charges", title="Smoker vs Charges")

    elif plot == "Region Distribution":
        dataset["region"].value_counts().plot(kind="bar", ax=ax, color="skyblue")
        ax.set(xlabel="Region", ylabel="Count", title="Region Distribution")

    elif plot == "Charges Distribution":
        ax.hist(dataset["charges"], bins=25, color="salmon", edgecolor="black")
        ax.set(xlabel="Charges", ylabel="Frequency", title="Distribution of Charges")

    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Model Details")
    info = available_models[selected_model_name]

    st.markdown(f"""
**Algorithm:** {selected_model_name}

**Target Variable:** Charges

**Features Used**
- Age
- Gender
- BMI
- Children
- Smoker
- Region
""")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", info["r2"])
    col2.metric("RMSE", info["rmse"])
    col3.metric("MAE", info["mae"])

    st.divider()
    st.subheader("Compare All Models")
    compare_df = pd.DataFrame([
        {"Model": name, "R² Score": m["r2"], "RMSE": m["rmse"], "MAE": m["mae"]}
        for name, m in available_models.items()
    ])
    st.dataframe(compare_df, use_container_width=True, hide_index=True)