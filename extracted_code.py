# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,

# THEN FEEL FREE TO DELETE THIS CELL.

# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON

# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR

# NOTEBOOK.

import kagglehub

ejlok1_toronto_emotional_speech_set_tess_path = kagglehub.dataset_download('ejlok1/toronto-emotional-speech-set-tess')



print('Data source import complete.')

import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib.pyplot as plt

import librosa

import librosa.display

from IPython.display import Audio

import warnings

warnings.filterwarnings('ignore')
paths = []

labels = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        paths.append(os.path.join(dirname, filename))

        label = filename.split('_')[-1]

        label = label.split('.')[0]

        labels.append(label.lower())

    if len(paths) == 2800:

        break

print('Dataset is Loaded')
len(paths)
paths[:5]
labels[:5]
## Create a dataframe

df = pd.DataFrame()

df['speech'] = paths

df['label'] = labels

df.head()
df['label'].value_counts()
sns.countplot(data=df, x='label')
def waveplot(data, sr, emotion):

    plt.figure(figsize=(10,4))

    plt.title(emotion, size=20)

    librosa.display.waveshow(data, sr=sr)

    plt.show()



def spectogram(data, sr, emotion):

    x = librosa.stft(data)

    xdb = librosa.amplitude_to_db(abs(x))

    plt.figure(figsize=(11,4))

    plt.title(emotion, size=20)

    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')

    plt.colorbar()
emotion = 'fear'

path = np.array(df['speech'][df['label']==emotion])[0]

data, sampling_rate = librosa.load(path)

waveplot(data, sampling_rate, emotion)

spectogram(data, sampling_rate, emotion)

Audio(path)
emotion = 'angry'

path = np.array(df['speech'][df['label']==emotion])[1]

data, sampling_rate = librosa.load(path)

waveplot(data, sampling_rate, emotion)

spectogram(data, sampling_rate, emotion)

Audio(path)
emotion = 'disgust'

path = np.array(df['speech'][df['label']==emotion])[0]

data, sampling_rate = librosa.load(path)

waveplot(data, sampling_rate, emotion)

spectogram(data, sampling_rate, emotion)

Audio(path)
emotion = 'neutral'

path = np.array(df['speech'][df['label']==emotion])[0]

data, sampling_rate = librosa.load(path)

waveplot(data, sampling_rate, emotion)

spectogram(data, sampling_rate, emotion)

Audio(path)
emotion = 'sad'

path = np.array(df['speech'][df['label']==emotion])[0]

data, sampling_rate = librosa.load(path)

waveplot(data, sampling_rate, emotion)

spectogram(data, sampling_rate, emotion)

Audio(path)
emotion = 'ps'

path = np.array(df['speech'][df['label']==emotion])[0]

data, sampling_rate = librosa.load(path)

waveplot(data, sampling_rate, emotion)

spectogram(data, sampling_rate, emotion)

Audio(path)
emotion = 'happy'

path = np.array(df['speech'][df['label']==emotion])[0]

data, sampling_rate = librosa.load(path)

waveplot(data, sampling_rate, emotion)

spectogram(data, sampling_rate, emotion)

Audio(path)
def extract_mfcc(filename):

    y, sr = librosa.load(filename, duration=3, offset=0.5)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    return mfcc
extract_mfcc(df['speech'][0])
X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X_mfcc
X = [x for x in X_mfcc]

X = np.array(X)

X.shape
## input split

X = np.expand_dims(X, -1)

X.shape
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

y = enc.fit_transform(df[['label']])
y = y.toarray()
y.shape
from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout



model = Sequential([

    LSTM(256, return_sequences=False, input_shape=(40,1)),

    Dropout(0.2),

    Dense(128, activation='relu'),

    Dropout(0.2),

    Dense(64, activation='relu'),

    Dropout(0.2),

    Dense(7, activation='softmax')

])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
# Train the model

history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)
# best val accuracy: 72.32

# use checkpoint to save the best val accuracy model

# adjust learning rate for slow convergence
epochs = list(range(50))

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



plt.plot(epochs, acc, label='train accuracy')

plt.plot(epochs, val_acc, label='val accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()

plt.show()
loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(epochs, loss, label='train loss')

plt.plot(epochs, val_loss, label='val loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()
model.save("emotion_model.keras")
import pickle



# Define the filename for the pickle file

pickle_filename = "emotion_model.pkl"



# Save the model to a pickle file

with open(pickle_filename, 'wb') as file:

    pickle.dump(model, file)



print(f"Model saved as {pickle_filename}")

def predict_emotion_from_audio(audio_file_path):

    # Preprocess the audio file

    processed_features = preprocess_audio_for_prediction(audio_file_path)



    # Expand dimensions for model input (batch size of 1)

    model_input = np.expand_dims(processed_features, axis=0)



    # Make prediction

    raw_predictions = loaded_model.predict(model_input)



    # Get the predicted label index

    predicted_label_index = np.argmax(raw_predictions)



    # Get the emotion labels from the OneHotEncoder

    emotion_labels = enc.categories_[0]



    # Get the predicted emotion label

    predicted_emotion = emotion_labels[predicted_label_index]



    return predicted_emotion



print("Function `predict_emotion_from_audio` defined.")
# Example usage:

# You would replace 'sample_audio_path' with the actual path to your uploaded audio file.

# For demonstration, we'll use the same sample audio as before.



# Assuming `sample_audio_path` is defined from previous steps

predicted_emotion_for_new_audio = predict_emotion_from_audio(sample_audio_path)

print(f"The predicted emotion for the audio file '{sample_audio_path}' is: {predicted_emotion_for_new_audio}")
def preprocess_audio_for_prediction(filename):

    y, sr = librosa.load(filename, duration=3, offset=0.5)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    # Reshape the MFCC features to (40, 1) to match the model's expected input shape for a single sample

    mfcc_reshaped = np.expand_dims(mfcc, axis=-1)

    return mfcc_reshaped



print("Function `preprocess_audio_for_prediction` defined.")
audio_file_to_predict = '/content/voice_audio.mp3'

predicted_emotion = predict_emotion_from_audio(audio_file_to_predict)

print(f"The predicted emotion for '{audio_file_to_predict}' is: {predicted_emotion}")
audio_file_to_predict_2 = '/content/voice_audio_1.mp3'

predicted_emotion_2 = predict_emotion_from_audio(audio_file_to_predict_2)

print(f"The predicted emotion for '{audio_file_to_predict_2}' is: {predicted_emotion_2}")
import tensorflow as tf

import pickle



# 1. Load the trained Keras model

# Using pickle.load() as emotion_model.keras was not found, but emotion_model.pkl was saved.

with open('emotion_model.pkl', 'rb') as file:

    loaded_model = pickle.load(file)

print("Model loaded successfully from pickle file.")



# 2. Select a sample audio file path

sample_audio_path = df['speech'][0]

print(f"Selected sample audio: {sample_audio_path}")



# 3. Preprocess this selected audio file

processed_audio_features = preprocess_audio_for_prediction(sample_audio_path)

print(f"Processed audio features shape: {processed_audio_features.shape}")



# 4. Expand the dimensions for batch prediction

model_input = np.expand_dims(processed_audio_features, axis=0)

print(f"Model input shape: {model_input.shape}")



# 5. Use the loaded_model.predict() method to get the raw predictions

raw_predictions = loaded_model.predict(model_input)

print("Raw predictions:")

print(raw_predictions)
from sklearn.preprocessing import OneHotEncoder

import numpy as np



# Get all unique labels from the original DataFrame

all_labels = np.array(df['label'].unique()).reshape(-1, 1)



# Re-initialize and fit the OneHotEncoder to ensure it has the correct categories

enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(all_labels)



predicted_label_index = np.argmax(raw_predictions)



# Get the emotion labels from the OneHotEncoder

emotion_labels = enc.categories_[0]



# Get the predicted emotion label

predicted_emotion = emotion_labels[predicted_label_index]



print(f"Predicted emotion: {predicted_emotion}")



# Print the actual emotion from the DataFrame for comparison

actual_emotion = df['label'][df['speech'] == sample_audio_path].iloc[0]

print(f"Actual emotion: {actual_emotion}")
