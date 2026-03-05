# Voice Emotion Detection 🎙️

A machine learning web application that detects human emotion from short voice audio clips. The app uses a trained Long Short-Term Memory (LSTM) network via Keras and is deployed locally using Streamlit.

## Features
- **Emotion Classification:** Detects emotions such as *angry, happy, neutral, sad, fear, disgust, and pleasant surprise (ps)*.
- **Audio Processing:** Uses `librosa` to extract Mel-Frequency Cepstral Coefficients (MFCCs) from `.wav` and `.mp3` audio files.
- **Clean UI:** Simple and interactive web interface built with Streamlit.

## Dependencies

The project requires **Python 3.12** or lower because TensorFlow does not currently support Python 3.13 or 3.14.

Core backend libraries:
* `streamlit`
* `tensorflow` / `keras`
* `librosa`
* `scikit-learn`
* `numpy`

## Project Structure
* `app.py`: The main Streamlit web application.
* `emotion_model.pkl`: The trained LSTM Keras model weights and architecture.
* `onehotencoder.pkl`: The scikit-learn OneHotEncoder fitted to the 7 emotion categories.
* `voice_emotion_detection.ipynb`: The original Jupyter Notebook containing data exploration, model training, and evaluation code.
* `extracted_code.py`: A Python script containing the extracted logic from the Jupyter Notebook.

## Setup and Installation

### Quick Start with `uv` (Recommended)
You can use [Astral's `uv`](https://github.com/astral-sh/uv) to automatically handle the Python 3.12 requirement and install all dependencies in a temporary environment:

```bash
uv run --python 3.12 --with streamlit --with librosa --with tensorflow --with scikit-learn --with numpy python -m streamlit run app.py
```

### Traditional Installation (Pip / Virtual Environments)
1. **Ensure you have Python 3.12 (or lower) installed.**
2. **Create a virtual environment:**
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate
```
3. **Install Dependencies:**
```bash
pip install streamlit tensorflow librosa scikit-learn numpy
```
4. **Run the Application:**
```bash
streamlit run app.py
```

## Usage
1. Once the application is running (by default on `http://localhost:8501`), open it in your browser.
2. Drag and drop any `.wav` or `.mp3` speech audio file into the file uploader.
3. You can click play to listen to the uploaded audio clip.
4. Click **Predict Emotion** to run the audio through the model and receive the prediction!
