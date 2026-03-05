import streamlit as st
import numpy as np
import librosa
import pickle
from keras.models import load_model

# Load model and encoder
@st.cache_resource
def load_assets():
    model = pickle.load(open('emotion_model.pkl', 'rb'))
    encoder = pickle.load(open('onehotencoder.pkl', 'rb'))
    return model, encoder

model, encoder = load_assets()

# Preprocessing function
def preprocess_audio_for_prediction(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc_reshaped = np.expand_dims(mfcc, axis=-1)
    return mfcc_reshaped

def predict_emotion(audio_file_path):
    processed_features = preprocess_audio_for_prediction(audio_file_path)
    model_input = np.expand_dims(processed_features, axis=0)
    raw_predictions = model.predict(model_input)
    predicted_label_index = np.argmax(raw_predictions)
    emotion_labels = [c[0] for c in encoder.categories_]
    predicted_emotion = emotion_labels[predicted_label_index]
    return predicted_emotion

# Streamlit App UI
st.set_page_config(page_title="Voice Emotion Detection", page_icon="🎙️")
st.title("Voice Emotion Detection 🎙️")
st.write("Upload an audio file (.wav, .mp3) to detect the underlying human emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    if st.button("Predict Emotion"):
        with st.spinner("Analyzing audio..."):
            try:
                emotion = predict_emotion(uploaded_file)
                st.success(f"Predicted Emotion: **{emotion.capitalize()}**")
                
                # Optional visual representation
                if emotion == 'happy':
                    st.balloons()
            except Exception as e:
                st.error(f"Error processing audio: {e}")
