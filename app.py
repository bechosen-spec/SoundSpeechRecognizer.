import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import sounddevice as sd
import wavio
import tempfile

# Function to predict class label for audio samples
def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]

# Load the trained model from the file
model = load_model("/home/boniface/Desktop/Projects/SoundSpeechRecognizer./best_model (1).hdf5")

# Define the list of classes
classes = ['_background_noise_', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

# Hide code output
st.set_option('deprecation.showfileUploaderEncoding', False)

# Create a Streamlit web app
st.title('Sound Speech Recognizer')

# Record voice command and display predicted text
if st.button('Record Voice'):
    st.text("Click the record button, speak your command, then click the stop button.")
    
    # Placeholder for displaying predicted text
    predicted_text_placeholder = st.empty()
    
    samplerate = 16000
    duration = 1  # seconds

    # Start recording audio
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()

    # Save the recorded audio as a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        temp_filename = temp_wav_file.name
        wavio.write(temp_filename, recording, samplerate)

    # Reading the voice command from the saved WAV file
    samples, sample_rate = librosa.load(temp_filename, sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)

    # Predict the text from the voice command
    predicted_text = predict(samples)

    # Display the predicted text
    predicted_text_placeholder.success(f"Predicted Text: {predicted_text}")
