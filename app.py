import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from prophet import Prophet

# RAG Components
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain.prompts import PromptTemplate
# New/Correct Import
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Weather Intelligence System",
    layout="wide",
    page_icon="🌧️"
)

st.title("🌦️ AI Weather Intelligence System (ML + LSTM + Prophet + RAG)")


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load("rain_prediction_model.pkl"), joblib.load("model_features.pkl")
    except:
        return None, None


@st.cache_resource
def load_lstm():
    try:
        return (
            tf.keras.models.load_model("rainfall_lstm_model.h5"),
            joblib.load("lstm_scaler.pkl"),
            joblib.load("lstm.pkl"),
        )
    except:
        return None, None, None


@st.cache_resource
def load_prophet():
    try:
        return joblib.load("rainfall_prophet_model.pkl")
    except:
        return None


ml_model, ml_features = load_ml_model()
lstm_model, lstm_scaler, lstm_pkl = load_lstm()
prophet_model = load_prophet()


# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
with st.sidebar:
    st.header("📌 Navigation")
    option = st.selectbox(
        "Choose a Module",
        [
            "🎯 Rain Prediction (ML)",
            "📈 Rainfall Forecasting (LSTM)",
            "🔮 Rainfall Forecasting (Prophet)",
            "🤖 Weather RAG Chatbot",
        ],
    )


# ------------------------------------------------------------
# 🎯 MODULE 1 — ML RAIN PREDICTION
# ------------------------------------------------------------
if option == "🎯 Rain Prediction (ML)":
    st.subheader("🌧️ Rain Prediction using Machine Learning")

    if ml_model is None:
        st.error("❌ Model not found! Upload `rain_prediction_model.pkl`")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            MinTemp = st.number_input("MinTemp", -10.0, 50.0, 15.0)
            MaxTemp = st.number_input("MaxTemp", -10.0, 50.0, 25.0)
            Rainfall = st.number_input("Rainfall", 0.0, 300.0, 2.0)
            Evap = st.number_input("Evaporation", 0.0, 30.0, 5.0)
            Sun = st.number_input("Sunshine", 0.0, 15.0, 7.0)
            Gust = st.number_input("WindGustSpeed", 0, 200, 45)

        with col2:
            WD9 = st.number_input("WindDir9am", 0, 360, 90)
            WD3 = st.number_input("WindDir3pm", 0, 360, 120)
            WS9 = st.number_input("WindSpeed9am", 0, 150, 10)
            WS3 = st.number_input("WindSpeed3pm", 0, 150, 15)
            H9 = st.slider("Humidity9am", 0, 100, 60)
            H3 = st.slider("Humidity3pm", 0, 100, 55)

        with col3:
            P9 = st.number_input("Pressure9am", 900.0, 1100.0, 1012.0)
            P3 = st.number_input("Pressure3pm", 900.0, 1100.0, 1008.0)
            C9 = st.slider("Cloud9am", 0, 100, 40)
            C3 = st.slider("Cloud3pm", 0, 100, 35)
            T9 = st.number_input("Temp9am", -10.0, 50.0, 20.0)
            T3 = st.number_input("Temp3pm", -10.0, 50.0, 28.0)

        user_row = [[MinTemp, MaxTemp, Rainfall, Evap, Sun, Gust, WD9, WD3,
                     WS9, WS3, H9, H3, P9, P3, C9, C3, T9, T3]]

        df_input = pd.DataFrame(user_row, columns=ml_features)

        if st.button("Predict"):
            pred = ml_model.predict(df_input)[0]
            st.success("🌧 Rain Likely!" if pred == 1 else "☀ No Rain Expected")


# ------------------------------------------------------------
# 📈 MODULE 2 — LSTM FORECAST
# ------------------------------------------------------------
if option == "📈 Rainfall Forecasting (LSTM)":
    st.subheader("📈 Rainfall Forecasting using LSTM")

    if lstm_model is None:
        st.error("❌ Missing LSTM model files.")
    else:
        file = st.file_uploader("Upload Rainfall CSV (Date, Rainfall)", type="csv")

        if file:
            df = pd.read_csv(file)
            df["Date"] = pd.to_datetime(df["Date"])

            seq = lstm_pkl["seq_len"]
            data = df["Rainfall"].values.reshape(-1, 1)
            scaled = lstm_scaler.transform(data)

            X = np.array([scaled[-seq:]]).reshape(1, seq, 1)

            pred_scaled = lstm_model.predict(X)
            forecast = lstm_scaler.inverse_transform(pred_scaled)[0][0]

            st.success(f"🌧 Forecast Rainfall (Tomorrow): {forecast:.2f} mm")


# ------------------------------------------------------------
# 🔮 MODULE 3 — PROPHET FORECAST
# ------------------------------------------------------------
if option == "🔮 Rainfall Forecasting (Prophet)":
    st.subheader("🔮 Prophet Forecast")

    if prophet_model is None:
        st.warning("❌ Prophet model not found! Upload forecast CSV instead.")
        up = st.file_uploader("Upload Prophet CSV", type="csv")
        if up:
            df = pd.read_csv(up)
            st.line_chart(df["yhat"])
    else:
        future = prophet_model.make_future_dataframe(periods=30)
        forecast = prophet_model.predict(future)

        st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))


# ------------------------------------------------------------
# 🤖 MODULE 4 — RAG CHATBOT
# ------------------------------------------------------------
if option == "🤖 Weather RAG Chatbot":
    st.subheader("🤖 AI Weather Chatbot")

    API = st.text_input("Enter Groq API Key", type="password")

    if API:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        # RAG Prompt
        RAG_PROMPT = """
        You are an AI Weather Assistant.
        Use the context to answer accurately.
        If info is missing, say "Not enough information."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        prompt = PromptTemplate(
            template=RAG_PROMPT,
            input_variables=["context", "question"]
        )

        llm = ChatGroq(
            groq_api_key=API,
            model_name="llama-3.3-70b-versatile"
        )

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        q = st.text_input("Ask a weather question:")

        if q:
            st.write(rag_chain.invoke(q))

