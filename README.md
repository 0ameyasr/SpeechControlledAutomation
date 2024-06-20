# Speech Controlled Automation

### Directory structure and details:
1. The code handling the logic for the ESP32 is located in automator/automator.ino and can be viewed using the Arduino IDE (Recommended version 2.3.x).
2. The fully trained model after the training process of the model is present as the fully_trained.model directory.
3. The model directory contains the code for data preprocessing (data_preprocessor.py) and training of the model (training_machine.py).
4. The dataset of 4,775 files that we used to train our model is present in the speech_data directory. It is classified into 7 subdirectories, 1 for noise and the rest of the 6 for the commands.
5. The predictor program is the application interface used to run our model. Simply executing and interacting with it will allow the user to give audio input, and check whether the predicted word is correct.
6. Three .npz files contain the results of the data preprocessing. Drivers and scripts can be written to inspect the data (computed spectrograms) in them.

### How to use the model:
1. To use the full benefit of the project, the hardware model would be required to be serially connected (Windows, COM ports) to the local machine running the model. 
2. After the microcontroller has been connected, the predictor.py program can be run. It will sample audio input in regular intervals and send predictions to the actuator logic.
3. You can choose to not run predictor.py with the hardware, and comment/remove certain dependencies and functions in the same.
4. Continue to interact with the predictor interface.
5. If you are using hardware, remember to upload the code automator.ino onto the microcontroller ESP-32 setup so that it can handle actuator logic.

### Note:
Even if you are not using the required hardware, simply running predictor.py would allow you to test the model. Beware, it can hallucinate if there is a lot of noise through your audio input.
