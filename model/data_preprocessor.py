import tensorflow
import numpy
import soundfile
from tensorflow.python.ops import gen_audio_ops
import matplotlib.pyplot
from tqdm import tqdm
import pickle

SAMPLING_RATE = 16000
NOISE_FLOOR = 0.1
MINIMUM_SPEECH_LENGTH = SAMPLING_RATE/4
COMMANDS = ['on','zero','one','two','three','off','_background']
TRAINING_SET = []
VALIDATION_SET = []
TESTING_SET = []
BACKGROUND_SET = []

TRAINING_SET_SIZE=0.8
VALIDATION_SET_SIZE=0.1
TESTING_SET_SIZE=0.1

"""
Audio Data-Preprocessing

This step ensures that the data used to train the model will be clean 
and fit for use. 
"""

#Retrieve all audio files from the dataset, for the specific command
def getFilesFromDataset(command:str):
    audio_dataset = tensorflow.io.gfile.join("speech_data",command)
    return tensorflow.io.gfile.glob(audio_dataset+"/*.wav")

#Get voice stamps for the audio, filter on the basis of noise_floor
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
    audio, _ = soundfile.read(file.numpy().decode('utf-8'))
    audio_tensor = tensorflow.convert_to_tensor(audio)
    audio = tensorflow.cast(audio_tensor[:], tensorflow.float32)
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

#Process files sequentially from the dataset
def processFilesFromDataset(file_names, label, repeat=1):
    file_names = tensorflow.repeat(file_names, repeat)
    return [(processFile(file_name), label) for file_name in tqdm(file_names, desc=f"({label})", leave=False)]

#Process commands from the speech_data directory
def processCommand(command,repeat=1):
    #Indexing and labelling of the word
    label = COMMANDS.index(command)
    file_names = [file_name for file_name in tqdm(getFilesFromDataset(command), desc=f"Processing.. for {command}", leave=False) if isFileValid(file_name)]
    
    numpy.random.shuffle(file_names)

    #Initialise the size of all sets
    training_set_size=int(TRAINING_SET_SIZE*len(file_names))
    validation_set_size=int(VALIDATION_SET_SIZE*len(file_names))
    testing_set_size=int(TESTING_SET_SIZE*len(file_names))

    #Randomly shuffle the filenames
    TRAINING_SET.extend(processFilesFromDataset(file_names[:training_set_size],label,repeat=repeat))
    VALIDATION_SET.extend(processFilesFromDataset(file_names[training_set_size:training_set_size+validation_set_size],label,repeat=repeat))
    TESTING_SET.extend(processFilesFromDataset(file_names[training_set_size+validation_set_size:],label,repeat=repeat))

#Process the background noise for fitting into the model
def processBackground(file_name, label):
    #Load the audio file
    audio,_ = soundfile.read(file_name)
    audio_tensor = tensorflow.convert_to_tensor(audio)
    audio = tensorflow.cast(audio_tensor[:],tensorflow.float32)
    audio_length = len(audio)
    samples = []

    for section_start in tqdm(range(0, audio_length-SAMPLING_RATE, 8000), desc=file_name, leave=False):
        section_end = section_start + SAMPLING_RATE
        section = audio[section_start:section_end]
        spectrogram = mapSpectrogram(section)
        samples.append((spectrogram,label))

    for section_index in tqdm(range(1000), desc='Simulated Words',leave=False):
        section_start = numpy.random.randint(0,audio_length-SAMPLING_RATE)
        section_end = section_start + SAMPLING_RATE
        section = numpy.reshape(audio[section_start:section_end], (SAMPLING_RATE))

        result = numpy.zeros((SAMPLING_RATE))
        voice_length = numpy.random.randint(MINIMUM_SPEECH_LENGTH/2,SAMPLING_RATE)
        voice_start = numpy.random.randint(0,SAMPLING_RATE-voice_length)
        
        hamming = numpy.hamming(voice_length)
        result[voice_start:voice_start+voice_length] = hamming * section[voice_start:voice_start+voice_length]
        spectrogram = mapSpectrogram(numpy.reshape(section, (16000,1)))
        samples.append((spectrogram,label))

    numpy.random.shuffle(samples)
    training_set_size = int(TRAINING_SET_SIZE*len(samples))
    validation_set_size = int(VALIDATION_SET_SIZE*len(samples))
    testing_set_size = int(TESTING_SET_SIZE*len(samples))

    TRAINING_SET.extend(samples[:training_set_size])
    VALIDATION_SET.extend(samples[training_set_size:training_set_size+validation_set_size])
    TESTING_SET.extend(samples[training_set_size+validation_set_size:])
    return samples

#Get audio files from the audio directory for the given command-word
command = "on"
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


#Process all words and all the files for pre-processing
for command in tqdm(COMMANDS,desc="Processing words..."):
    if '_' not in command:
        processCommand(command,repeat=1)
print(len(TRAINING_SET),len(TESTING_SET),len(VALIDATION_SET))

#Process background noise
for file_name in tqdm(getFilesFromDataset('_background_noise_'),desc="Processing background noise..."):
        processBackground(file_name,COMMANDS.index("_background"))
print(len(TRAINING_SET),len(TESTING_SET),len(VALIDATION_SET))

background_data = []
for file_name in tqdm(getFilesFromDataset('_background_noise_'), desc="Processing background noise..."):
  background_data.extend(processBackground(file_name, COMMANDS.index("_background")))

# Shuffle background data
numpy.random.shuffle(background_data)

# Choose a portion for training (adjust portion as needed)
background_training_size = int(0.25 * len(background_data))
background_training_data = background_data[:background_training_size]

# Append background data to training set
BACKGROUND_SET.extend(background_training_data)

#Perform culling on all non-homogenously shaped spectrograms
def filterSpectrogramShapes(spectrograms):
    filtered_spectrograms = [(spec, label) for spec, label in spectrograms if len(spec.shape) == 2]
    return filtered_spectrograms

# Filter the sets
TRAINING_SET = filterSpectrogramShapes(TRAINING_SET)
VALIDATION_SET = filterSpectrogramShapes(VALIDATION_SET)
TESTING_SET = filterSpectrogramShapes(TESTING_SET)
BACKGROUND_SET = filterSpectrogramShapes(BACKGROUND_SET)

# #Export the pre-processed data to sets
X_train, Y_train = zip(*TRAINING_SET)
X_validate, Y_validate = zip(*VALIDATION_SET)
X_test, Y_test = zip(*TESTING_SET)
X_back, Y_back = zip(*BACKGROUND_SET)

import numpy as np

# Function to pad or trim arrays to a fixed shape (target_shape)
def preprocess_arrays(arrays, target_shape):
    preprocessed_arrays = []
    for arr in arrays:
        if arr.shape != target_shape:
            if arr.shape[0] < target_shape[0]:
                # Pad the array if it is smaller than the target shape
                pad_width = [(0, target_shape[0] - arr.shape[0])] + [(0, 0)] * (len(arr.shape) - 1)
                padded_arr = np.pad(arr, pad_width, mode='constant')
                preprocessed_arrays.append(padded_arr)
            else:
                # Trim the array if it is larger than the target shape
                trim_start = (arr.shape[0] - target_shape[0]) // 2
                trim_end = trim_start + target_shape[0]
                trimmed_arr = arr[trim_start:trim_end]
                preprocessed_arrays.append(trimmed_arr)
        else:
            # Array already has the target shape, no processing needed
            preprocessed_arrays.append(arr)
    return preprocessed_arrays

# Define the target shape for your spectrograms
target_shape = (124, 129)


# Preprocess the arrays to ensure homogeneous shapes
X_train_processed = preprocess_arrays(X_train, target_shape)
X_validate_processed = preprocess_arrays(X_validate, target_shape)
X_test_processed = preprocess_arrays(X_test, target_shape)
X_back_processed = preprocess_arrays(X_back, target_shape)

# Save the preprocessed data
np.savez_compressed(
    "training_spectrogram.npz",
    X=X_train_processed, Y=Y_train)
print("Saved training data")
np.savez_compressed(
    "validation_spectrogram.npz",
    X=X_validate_processed, Y=Y_validate)
print("Saved validation data")
np.savez_compressed(
    "test_spectrogram.npz",
    X=X_test_processed, Y=Y_test)
print("Saved test data")

# Print the labels present in the training set, all that are possible
training_labels = set(Y_train)
print("Labels present in the training set:")
for label in training_labels:
    print(f"Label {label}: {COMMANDS[label]}")

# Plot the first spectrogram from the training set, to check if it was preprocessed
first_spectrogram = X_train[0]
label = Y_train[0]
plotSpectrogram(first_spectrogram, title=f"First Spectrogram in Training Set - Label: {COMMANDS[label]}")