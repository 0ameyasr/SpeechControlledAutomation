# Import all of the dependencies
import datetime
import tensorflow as tf
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.data import Dataset
from tinymlgen import port

#List of the command words in categorical order
command_words = ['on','zero','one','two','three','off','_background']

#Load up the sprectrograms and labels
test_spectrogram = np.load('test_spectrogram.npz')
validation_spectrogram = np.load('validation_spectrogram.npz')
training_spectrogram = np.load('training_spectrogram.npz')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) #This is for CoLab

#Extract the data from the files into datasets
X_train = training_spectrogram['X']
Y_train_cats = training_spectrogram['Y']
X_validate = validation_spectrogram['X']
Y_validate_cats = validation_spectrogram['Y']
X_test = test_spectrogram['X']
Y_test_cats = test_spectrogram['Y']

#Get the width and height of the spectrogram images for consistent output
IMG_WIDTH=X_train[0].shape[0]
IMG_HEIGHT=X_train[0].shape[1]

#Plot a distribution of the words
plt.hist(Y_train_cats, bins=range(0,len(command_words)+1), align='left')

#Print all of the categories mentioned in the datasets
unique, counts = np.unique(Y_train_cats, return_counts=True)
print(unique, counts)
dict(zip([command_words[i] for i in unique], counts))

#Perform One-Hot Encoding on the datasets to label and encode our data
Y_train = tf.one_hot(Y_train_cats, len(command_words))
Y_validate = tf.one_hot(Y_validate_cats, len(command_words))
Y_test = tf.one_hot(Y_test_cats, len(command_words))

#Create the datasets for training
batch_size = 16
train_dataset = Dataset.from_tensor_slices(
    (X_train, Y_train)
).repeat(
    count=-1
).shuffle(
    len(X_train)
).batch(
    batch_size
)
validation_dataset = Dataset.from_tensor_slices((X_validate, Y_validate)).batch(X_validate.shape[0]//10)
test_dataset = Dataset.from_tensor_slices((X_test, Y_test)).batch(len(X_test))

#Model our neural network
model = Sequential([
    Conv2D(4, 3,
           padding='same',
           activation='relu',
           kernel_regularizer=regularizers.l2(0.001),
           name='conv_layer1',
           input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    MaxPooling2D(name='max_pooling1', pool_size=(2,2)),
    Conv2D(4, 3,
           padding='same',
           activation='relu',
           kernel_regularizer=regularizers.l2(0.001),
           name='conv_layer2'),
    MaxPooling2D(name='max_pooling3', pool_size=(2,2)),
    Flatten(),
    Dropout(0.1),
    Dense(
        80,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='hidden_layer1'
    ),
    Dropout(0.1),
    Dense(
        len(command_words),
        activation='softmax',
        kernel_regularizer=regularizers.l2(0.001),
        name='output'
    )
])
model.summary()

#Train the model for 50 epochs
epochs=50
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.save("trained.model")

#Test our trained model
import keras
import itertools
model2 = keras.models.load_model("trained.model")
results = model2.evaluate(X_test, tf.cast(Y_test, tf.float32), batch_size=128)
predictions = model2.predict(X_test, 128)

#Plot the confusion matrix to see how the model performs
def plot_confusion_matrix(cm, class_names):
    #Normalise the matrix
    cm = cm.numpy()
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    #Get the figure, and plot the confusion matrix
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    #Format, colorize the matrix
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

cm = tf.math.confusion_matrix(
    labels=tf.argmax(Y_test, 1), predictions=tf.argmax(predictions, 1)
)

# Get most probable prediction, print them too
predicted_indices = np.argmax(predictions, axis=1)
predicted_words = [command_words[idx] for idx in predicted_indices]
print(predicted_words)
plot_confusion_matrix(cm, command_words)

batch_size = 30
complete_train_X = np.concatenate((X_train, X_validate, X_test))
complete_train_Y = np.concatenate((Y_train, Y_validate, Y_test))
complete_train_dataset = Dataset.from_tensor_slices((complete_train_X, complete_train_Y)).repeat(count=-1).shuffle(len(complete_train_X)).batch(batch_size)
history = model2.fit(
    complete_train_dataset,
    steps_per_epoch=len(complete_train_X) // batch_size,
    epochs=5
)
model2.save("fully_trained.model")

results = model2.evaluate(complete_train_X, tf.cast(complete_train_Y, tf.float32), batch_size=128)
predictions = model2.predict(complete_train_X, 128)
cm = tf.math.confusion_matrix(
    labels=tf.argmax(complete_train_Y, 1), predictions=tf.argmax(predictions, 1)
)
plot_confusion_matrix(cm, command_words)

#Print predictions from the fully trained model now
import numpy as np
predicted_indices = np.argmax(predictions, axis=1)
predicted_words = [command_words[idx] for idx in predicted_indices]
print(predicted_words)
