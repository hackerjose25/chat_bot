import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ollama

# ==============================
# PAGE SETUP
# ==============================
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺")
st.title("🩺 JOSE AI Assistant")
st.caption("This AI is not a doctor. Always consult a professional.")

# ==============================
# LOAD ML MODEL (cached)
# ==============================
@st.cache_resource
def load_model():
    train = pd.read_csv("training_data.csv")
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]

    X = train.drop("prognosis", axis=1)
    y = train["prognosis"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, X.columns.tolist()

model, symptoms_list = load_model()

# ==============================
# SESSION STATE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# DISPLAY CHAT HISTORY
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# USER INPUT
# ==============================
if prompt := st.chat_input("Describe your symptoms..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    response = ollama.chat(
        model="llama3",
        messages=st.session_state.messages[-4:]
    )

    bot_reply = response["message"]["content"]

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    # ==============================
    # Disease Prediction
    # ==============================
    user_symptoms = [
        s.strip() for s in prompt.lower().split(",")
        if s.strip() in symptoms_list
    ]

    if user_symptoms:
        input_data = [0] * len(symptoms_list)

        for symptom in user_symptoms:
            index = symptoms_list.index(symptom)
            input_data[index] = 1

        input_df = pd.DataFrame([input_data], columns=symptoms_list)
        probabilities = model.predict_proba(input_df)[0]
        top_index = np.argmax(probabilities)

        disease = model.classes_[top_index]
        confidence = round(probabilities[top_index] * 100, 2)

        st.warning(f"🩺 Possible Condition: {disease}")
        st.info(f"Confidence: {confidence}%")