#Importing all dependencies of the project
import serial
import pyaudio
import wave
import os
import soundfile
import tensorflow as tf
from model.data_preprocessor import mapSpectrogram
import time
from tqdm import tqdm

#Load our trained model, set the command words and the messenger to communicate with ESP-32
model = tf.keras.models.load_model('fully_trained.model')
messenger = serial.Serial('COM3',baudrate=9600) #Comment this line, if you are not using the hardware model
command_words = ['on','zero','one','two','three','off','_background']

#Record audio temporarily for one-second sample
def record_audio(duration=2, filename=None):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    if filename is None:
        filename = "temp_audio.wav"
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, filename)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wave_file = wave.open(file_path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()
    print("Captured audio.")
    return file_path

#Preprocess the audio to return a spectrogram
def preprocess_audio(file_path):
    audio, _ = soundfile.read(file_path)
    spectrogram = mapSpectrogram(audio)
    if len(spectrogram.shape) == 1:
        spectrogram = tf.expand_dims(spectrogram, axis=0)
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
    elif len(spectrogram.shape) == 2:
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.image.resize(spectrogram, (124, 129))
    spectrogram = (spectrogram - tf.reduce_mean(spectrogram)) / tf.math.reduce_std(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=0)    
    return spectrogram

#Predictor loop
def automator():
    while True:
        print("\nReady to record audio! ")
        time.sleep(1)
        print("-----------BEGIN--------------")
        temp_wav_file = record_audio()
        time.sleep(1)
        print("------------STOP--------------")

        processed_audio = preprocess_audio(temp_wav_file)
        predictions = model.predict(processed_audio)
        predicted_index = tf.argmax(predictions, axis=1).numpy()[0]
        predicted_word = command_words[predicted_index]
        print(f"I heard: {predicted_word}")
        messenger.write(bytes(predicted_word,'utf-8'))
        time.sleep(5)

#Main driver
if __name__=="__main__":
    print("\nSpeech Controlled Automation")
    print(" 'on' to turn on ESP32's internal LED, and process other commands")
    print(" 'off' to turn off ESP32's internal LED, and stop processing other commands")
    print(" 'zero' to turn on the red LED")
    print(" 'one' to turn on the yellow LED")
    print(" 'two' to turn on the orange LED")
    print(" 'three' to turn on the green LED\n")

    automator()