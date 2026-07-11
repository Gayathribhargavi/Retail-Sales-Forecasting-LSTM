import streamlit as st
import pandas as pd
import numpy as np
import os 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/combo-chart--v1.png",
    width=80
)
st.sidebar.title("Retail Sales Forecasting")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📈 Predict Sales",
        "📊 Forecast Dashboard",
        "📋 Dataset Explorer",
        "ℹ️ About Project"
    ]
)
# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":

    st.title("📈 Retail Sales Forecasting Using LSTM")
    st.markdown("---")
    st.write("""
Welcome to the **Retail Sales Forecasting System**.
This project predicts future retail sales using **Long Short-Term Memory (LSTM)**,
which is an advanced **Recurrent Neural Network (RNN)**.
The model learns historical sales patterns and predicts upcoming weekly sales.
This application was developed using **Python, TensorFlow, Streamlit, Plotly,
Pandas and Scikit-Learn**.
""")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "LSTM")
    col2.metric("Model Type", "RNN")
    col3.metric("Framework", "TensorFlow")
    st.markdown("---")
    st.subheader("✨ Features")
    st.write("✅ Predict Future Weekly Sales")
    st.write("✅ Forecast Dashboard")
    st.write("✅ Interactive Graph")
    st.write("✅ Dataset Explorer")
    st.write("✅ Download Forecast CSV")
    st.success("Project Loaded Successfully 🚀")
# -----------------------------
# PREDICT PAGE
# -----------------------------
elif page == "📈 Predict Sales":

    st.title("📈 Predict Future Weekly Sales")

    st.markdown("### Enter the last 10 weekly sales values")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(BASE_DIR, "..", "Models", "lstm_sales_forecasting.keras")
    model = load_model(MODEL_PATH)

    DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "train.csv")
    train = pd.read_csv(DATA_PATH)

    sales = train["Weekly_Sales"].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaler.fit(sales)

    st.write("You can edit the values below and click Predict.")

    default_values = sales[-10:].flatten()

    user_input = []

    col1, col2 = st.columns(2)

    for i in range(10):

        if i < 5:

            value = col1.number_input(
                f"Week {i+1}",
                value=float(default_values[i]),
                step=100.0
            )

        else:

            value = col2.number_input(
                f"Week {i+1}",
                value=float(default_values[i]),
                step=100.0
            )

        user_input.append(value)

    if st.button("🔮 Predict Next Week Sales"):

        input_array = np.array(user_input).reshape(-1,1)

        scaled_input = scaler.transform(input_array)

        X = scaled_input.reshape(1,10,1)

        prediction = model.predict(X)

        predicted_sales = scaler.inverse_transform(prediction)

        st.success(
            f"Predicted Weekly Sales = ₹ {predicted_sales[0][0]:,.2f}"
        )

        st.balloons()

# -----------------------------
# DASHBOARD
# -----------------------------
# -----------------------------
# FORECAST DASHBOARD
# -----------------------------
elif page == "📊 Forecast Dashboard":

    import pandas as pd
    import plotly.express as px

    st.title("📊 Retail Sales Forecast Dashboard")

    st.markdown("---")

    # Load Forecast CSV
    try:
        FORECAST_PATH = os.path.join(
            BASE_DIR,
            "..",
            "Forecast",
            "future_sales_forecast.csv"
        )
        forecast = pd.read_csv(FORECAST_PATH)

    except FileNotFoundError:
        st.error("Forecast file not found!")
        st.stop()

    # Show Dataset
    st.subheader("Forecast Data")

    st.dataframe(forecast, use_container_width=True)

    st.markdown("---")

    # KPIs
    st.subheader("Forecast Summary")

    sales_column = forecast.columns[-1]

    total_sales = forecast[sales_column].sum()
    average_sales = forecast[sales_column].mean()
    maximum_sales = forecast[sales_column].max()
    minimum_sales = forecast[sales_column].min()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Forecast",
        f"₹ {total_sales:,.0f}"
    )

    col2.metric(
        "Average Sales",
        f"₹ {average_sales:,.0f}"
    )

    col3.metric(
        "Maximum Sales",
        f"₹ {maximum_sales:,.0f}"
    )

    col4.metric(
        "Minimum Sales",
        f"₹ {minimum_sales:,.0f}"
    )

    st.markdown("---")

    # Line Chart
    st.subheader("📈 Forecast Trend")

    x_column = forecast.columns[0]

    fig = px.line(
        forecast,
        x=x_column,
        y=sales_column,
        markers=True,
        title="Future Weekly Sales Forecast"
    )

    fig.update_layout(
        xaxis_title="Weeks",
        yaxis_title="Predicted Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # Bar Chart
    st.subheader("📊 Sales Comparison")

    fig2 = px.bar(
        forecast,
        x=x_column,
        y=sales_column,
        text=sales_column
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    # Download CSV

    csv = forecast.to_csv(index=False)

    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="future_sales_forecast.csv",
        mime="text/csv"
    )

    st.success("Forecast Dashboard Loaded Successfully")

# -----------------------------
# DATASET
# -----------------------------
# -----------------------------
# DATASET EXPLORER
# -----------------------------
elif page == "📋 Dataset Explorer":

    import pandas as pd
    import plotly.express as px

    st.title("📋 Dataset Explorer")

    st.markdown("---")

    # Load Dataset
    try:
        TRAIN_PATH = os.path.join(
            BASE_DIR,
            "..",
            "Dataset",
            "train.csv"
        )
        train = pd.read_csv(TRAIN_PATH)
    except FileNotFoundError:
        st.error("train.csv not found!")
        st.stop()

    st.success("Dataset Loaded Successfully")

    st.markdown("---")

    # Dataset Shape
    st.subheader("📊 Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", train.shape[0])
    col2.metric("Columns", train.shape[1])

    st.markdown("---")

    # Dataset Preview
    st.subheader("📄 Dataset Preview")

    rows = st.slider(
        "Select number of rows",
        min_value=5,
        max_value=50,
        value=10
    )

    st.dataframe(train.head(rows), use_container_width=True)

    st.markdown("---")

    # Select Columns
    st.subheader("📑 View Selected Columns")

    selected_columns = st.multiselect(
        "Choose Columns",
        train.columns,
        default=list(train.columns)
    )

    if selected_columns:
        st.dataframe(train[selected_columns])

    st.markdown("---")

    # Missing Values
    st.subheader("❓ Missing Values")

    missing = train.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    st.dataframe(missing)

    st.markdown("---")

    # Statistical Summary
    st.subheader("📈 Statistical Summary")

    st.dataframe(train.describe())

    st.markdown("---")

    # Correlation Heatmap
    st.subheader("🔥 Correlation Matrix")

    corr = train.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Histogram
    st.subheader("📊 Distribution of Weekly Sales")

    fig2 = px.histogram(
        train,
        x="Weekly_Sales",
        nbins=40,
        title="Weekly Sales Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Search Store
    st.subheader("🔍 Search Store")

    store = st.selectbox(
        "Select Store",
        sorted(train["Store"].unique())
    )

    filtered = train[train["Store"] == store]

    st.write(filtered)

    st.markdown("---")

    # Download Dataset

    csv = train.to_csv(index=False)

    st.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name="train.csv",
        mime="text/csv"
    )

    st.success("Dataset Explorer Loaded Successfully ✅")

# -----------------------------
# ABOUT
# -----------------------------
# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Retail Sales Forecasting Project")

    st.markdown("---")

    st.header("📌 Project Title")

    st.success(
        "Retail Sales Forecasting Using Long Short-Term Memory (LSTM) Recurrent Neural Network"
    )

    st.markdown("---")

    st.header("🎯 Project Objective")

    st.write("""
The objective of this project is to predict future retail sales using
historical weekly sales data.

This helps businesses estimate future demand, improve inventory planning,
optimize staffing, and make better business decisions.
""")

    st.markdown("---")

    st.header("🧠 About LSTM")

    st.write("""
LSTM (Long Short-Term Memory) is a special type of
Recurrent Neural Network (RNN).

Unlike traditional neural networks, LSTM remembers
past information for a long period using memory cells
and gates.

It is widely used for

• Time Series Forecasting

• Stock Price Prediction

• Weather Forecasting

• Sales Forecasting

• Speech Recognition

• Natural Language Processing
""")

    st.markdown("---")

    st.header("🏗 Project Workflow")

    st.write("""
Dataset

⬇

Data Preprocessing

⬇

Feature Scaling

⬇

Create Time Sequences

⬇

Train LSTM Model

⬇

Model Evaluation

⬇

Sales Prediction

⬇

Future Sales Forecast

⬇

Streamlit Dashboard
""")

    st.markdown("---")

    st.header("🛠 Technologies Used")

    tech = {
        "Technology":[
            "Python",
            "TensorFlow",
            "Keras",
            "Pandas",
            "NumPy",
            "Scikit-Learn",
            "Plotly",
            "Streamlit"
        ],
        "Purpose":[
            "Programming Language",
            "Deep Learning",
            "Neural Network API",
            "Data Processing",
            "Numerical Computing",
            "Data Preprocessing",
            "Visualization",
            "Web Application"
        ]
    }

    import pandas as pd

    st.table(pd.DataFrame(tech))

    st.markdown("---")

    st.header("📂 Dataset Information")

    st.write("""
Dataset Name :
Walmart Retail Sales Dataset

Files Used

✔ train.csv

✔ test.csv

✔ stores.csv

✔ features.csv
""")

    st.markdown("---")

    st.header("📊 Model Details")

    st.write("""
Algorithm

✔ Long Short-Term Memory (LSTM)

Model Category

✔ Deep Learning

Network Type

✔ Recurrent Neural Network (RNN)

Forecast Type

✔ Time-Series Forecasting
""")

    st.markdown("---")

    st.header("⭐ Project Features")

    st.write("""
✔ Predict Future Weekly Sales

✔ Interactive Dashboard

✔ Forecast Visualization

✔ Download Forecast CSV

✔ Dataset Explorer

✔ Deep Learning Model

✔ Professional Streamlit Interface
""")

    st.markdown("---")

    st.header("📈 Business Applications")

    st.write("""
Retail Industry

Inventory Management

Demand Forecasting

Warehouse Planning

Supply Chain Management

Business Analytics

Sales Strategy

Revenue Planning
""")

    st.markdown("---")

    st.header("👨‍💻 Developer")

    st.info("""
Developed By

Gayathri Bhargavi

B.Tech – Computer Science Engineering (AI)

Deep Learning Project
""")

    st.markdown("---")

    st.success("Retail Sales Forecasting Project Completed Successfully 🎉")