import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import ollama
import streamlit.components.v1 as components
from pathlib import Path
import math

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="Healthcare Assistant", page_icon="🩺", layout="wide")
st.html(Path(__file__).with_name("ui.css").read_text(encoding="utf-8"))
rays = "".join(
    f'<line x1="{26 + 10.4 * math.cos(a):.3f}" y1="{26 + 10.4 * math.sin(a):.3f}" '
    f'x2="{26 + 22.6 * math.cos(a):.3f}" y2="{26 + 22.6 * math.sin(a):.3f}" />'
    for a in (i * math.tau / 24 - math.pi / 2 for i in range(24))
)
st.html(f'''<header class="topbar" id="home">
    <a class="brand" href="#home" aria-label="Healthcare Assistant home">
    <svg viewBox="0 0 52 52" aria-hidden="true"><g stroke="#f4f4f4" stroke-width="1.4" stroke-linecap="round">{rays}</g><circle cx="26" cy="26" r="7.4" fill="#fbfbfb"/></svg>
    <span>Healthcare Assistant</span></a>
    <nav aria-label="Main navigation"><a href="#assistant">Assistant</a><a href="#consultation">Consult a doctor</a></nav>
    <a class="start-button" href="#assistant">Start Free <span aria-hidden="true">↗</span></a>
    </header>
    <section class="hero"><p class="eyebrow">YOUR HEALTH, A CLEARER CONVERSATION</p>
    <h1><span>Think clearly.</span><span>Decide confidently</span></h1>
    <p class="subtitle">An AI healthcare assistant that helps you understand<br>your symptoms and explore your next steps.</p>
    </section><div id="assistant" class="section-anchor"></div>''')
st.caption("This AI is not a doctor. Always consult a professional.")

# ==============================
# LOAD ML MODEL
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
# USER INPUT & CHAT
# ==============================

if prompt := st.chat_input("Describe your symptoms..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Thinking…"):
            response = ollama.chat(
                model="phi3",   # lighter model
                messages=st.session_state.messages[-2:]  # reduce memory usage
            )
    except ConnectionError:
        st.error("The AI assistant is temporarily unavailable. Start Ollama and send your message again. Symptom checks and doctor consultation are still available.")
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            st.error("The phi3 AI model is not installed. Run `ollama pull phi3`, then send your message again.")
        else:
            st.error("Ollama could not complete this response. Please try again. Symptom checks and doctor consultation are still available.")
    else:
        bot_reply = response["message"]["content"]
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

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

# ==============================
# DOCTOR CONSULTATION
# ==============================

st.html('''<section class="consultation-heading" id="consultation">
    <p class="eyebrow">A HUMAN WHEN YOU NEED ONE</p>
    <h2>Built for human care.<br>Supported by intelligence.</h2>
    <p>Need professional help? Connect with a doctor for a conversation.</p>
    </section>''')

if st.button("📞 Consult Doctor"):
    room_name = "healthcare-demo-room"
    components.iframe(
        f"https://meet.jit.si/{room_name}",
        height=600
    )
