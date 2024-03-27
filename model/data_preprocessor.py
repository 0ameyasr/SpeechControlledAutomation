import tensorflow
import numpy
import soundfile
from tensorflow.python.ops import gen_audio_ops
import matplotlib.pyplot
from tqdm import tqdm

SAMPLING_RATE = 16000
NOISE_FLOOR = 0.1
MINIMUM_SPEECH_LENGTH = SAMPLING_RATE/4
COMMANDS = ['zero','one','two','three','forward','backward','left','right']

"""
Audio Data-Preprocessing

This step ensures that the data used to train the model will be clean 
and fit for use. 
"""

#Retrieve all audio files from the dataset, for the specific command
def getFilesFromDataset(command:str):
    audio_dataset = tensorflow.io.gfile.join("speech_data",command)
    return tensorflow.io.gfile.glob(audio_dataset+"/*.wav")

def getVoiceStamps(audio: tensorflow.Tensor, noise_floor: float = NOISE_FLOOR):
    audio -= tensorflow.reduce_mean(audio)
    audio /= tensorflow.reduce_max(tensorflow.abs(audio))

    start = 0
    end = len(audio)
    
    # Find the start of voice stamp
    for i, sample in enumerate(audio):
        if numpy.abs(sample) > noise_floor:
            start = i
            break
    
    # Find the end of voice stamp
    for i in range(len(audio) - 1, -1, -1):
        if numpy.abs(audio[i]) > noise_floor:
            end = i
            break
    
    # Restrict the audio sample to one second
    expected_samples = 16000  # Assuming 16KHz sample rate for 1 second audio
    if end - start < expected_samples:
        pad_length = expected_samples - (end - start)
        pad_start = max(0, start - pad_length // 2)
        pad_end = min(len(audio), end + pad_length // 2)
        return audio[pad_start:pad_end + 1]
    else:
        return audio[start:end + 1]

#Retrieve the length of the voice segment from the audio
def getVoiceLength(audio:tensorflow.Tensor,noise_floor=NOISE_FLOOR):
    return len(audio)

#Check whether the voice present is detectable
def isAudible(audio:tensorflow.Tensor,required_length,noise_floor=NOISE_FLOOR):
    voice_length = getVoiceLength(audio)
    return voice_length >= required_length

#Check if the audio is of the expected, correct length
def isCorrectLength(audio:tensorflow.Tensor,expected_length):
    return audio.shape[0] == expected_length

#Check if the file is valid or not
def isFileValid(file:str):
    try:
        audio, _ = soundfile.read(file)
        if not isCorrectLength(audio,SAMPLING_RATE):
            return False
        audio = getVoiceStamps(audio.copy(),NOISE_FLOOR)
        if not isAudible(audio,MINIMUM_SPEECH_LENGTH,NOISE_FLOOR):
            return False
        audio=tensorflow.cast(audio[:],tensorflow.float32)
        return True
    except FileNotFoundError:
        print(f"File not found: {file}")
        return False

#Visualise the audio waveforms to check whether the audio is being registered
def visualizeAudio(audio, title="Audio Waveform"):
    matplotlib.pyplot.figure(figsize=(10, 3))
    matplotlib.pyplot.plot(audio)
    matplotlib.pyplot.xlabel("Time (samples)")
    matplotlib.pyplot.ylabel("Amplitude")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.show()

#Map the spectrogram from the audio file read
def mapSpectrogram(audio):
    # Convert audio to spectrogram
    spectrogram = tensorflow.signal.stft(audio, frame_length=256, frame_step=128)
    spectrogram = tensorflow.abs(spectrogram)

    # Convert to log scale
    log_spectrogram = tensorflow.math.log(spectrogram + 1e-6)

    # Normalize the data
    normalized_spectrogram = (log_spectrogram - tensorflow.math.reduce_mean(log_spectrogram)) / tensorflow.math.reduce_std(log_spectrogram)

    return normalized_spectrogram

#Visualise the mapped spectrogram
def plotSpectrogram(spectrogram, title="Spectrogram"):
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.imshow(spectrogram.numpy().T, aspect='auto', origin='lower')
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.xlabel("Time")
    matplotlib.pyplot.ylabel("Frequency")
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()

#Load normalised audio into the given sound files
def loadNormalisedAudio(file):
    audio,_ = soundfile.read(file)
    audio_tensor = tensorflow.convert_to_tensor(audio)
    audio = tensorflow.cast(audio_tensor[:],tensorflow.float32)
    audio -= tensorflow.reduce_mean(audio)
    audio /= tensorflow.reduce_max(tensorflow.abs(audio))
    return audio

#Add randomisation to the sound
def getRandomOffset(audio):
    # Calculate the end gap after trimming
    end_gap = len(audio) - MINIMUM_SPEECH_LENGTH
    
    # Calculate random offset within the end gap
    random_offset = numpy.random.uniform(0, end_gap)
    return random_offset

#Add some noise for real-time ambiguation
def addBackgroundNoise(audio, background_tensor, background_start, background_volume):
    # Calculate the length difference between audio and background noise
    audio_length = len(audio)
    background_length = len(background_tensor)
    length_diff = background_length - audio_length

    if length_diff > 0:
        # Background noise is longer, so crop it
        background = background_tensor[background_start:background_start + audio_length]
    elif length_diff < 0:
        # Background noise is shorter, so pad it with zeros
        padding = -length_diff
        background = tensorflow.pad(background_tensor[background_start:], [[0, padding]], 'CONSTANT')
    else:
        # Lengths are equal, use background noise as is
        background = background_tensor[background_start:]

    background = tensorflow.cast(background, tensorflow.float32)
    background = background - tensorflow.reduce_mean(background)
    background = background / tensorflow.reduce_max(tensorflow.abs(background))

    audio = audio + background_volume * background
    return audio

#Process file directly from the sound file in the dataset
def processFile(file_path):
    # Load and normalize audio
    audio = loadNormalisedAudio(file_path)
    
    # Get voice stamps and calculate random offset
    audio_stamps = getVoiceStamps(audio, NOISE_FLOOR)
    random_offset = getRandomOffset(audio_stamps)
    
    # Apply random offset to the audio stamps
    audio_stamps = numpy.roll(audio_stamps, -int(random_offset))
    
    # Add background noise
    background_files = getFilesFromDataset('_background_noise_')
    background_file = numpy.random.choice(background_files)
    background_audio, _ = soundfile.read(background_file)
    background_tensor = tensorflow.convert_to_tensor(background_audio)
    background_start = numpy.random.randint(0, len(background_tensor) - SAMPLING_RATE)
    background_volume = numpy.random.uniform(0, 0.1)
    audio_stamps = addBackgroundNoise(audio_stamps, background_tensor, background_start, background_volume)
    
    return mapSpectrogram(audio_stamps)


#Get audio files from the audio directory for the given command-word
command = "cat"
audio_directory = getFilesFromDataset(command)
for file in audio_directory:
    if isFileValid(file):
        audio,_ = soundfile.read(file)
        plotSpectrogram(mapSpectrogram(audio),"Audio without ambiguation")
        plotSpectrogram(processFile(file))
        visualizeAudio(audio,f"Valid {file}")
        print(f"Valid: {file}")
    else:
        print(f"Invalid: {file}")