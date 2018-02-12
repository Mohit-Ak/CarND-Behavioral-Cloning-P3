from util import getImagePathsAndCorrectedMeasurements
from util import generator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#for model
import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

			
def loadNvidiaModel():
    """
   c
    @returns: NVIDIA CNN model
    """
    #Declaring a sequential model
    model = Sequential()
    # Normalizing the input images
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Cropping the input as the sky  does not impact the steering angles
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    # CNN 1 + Rectified Linear Unit + Subsampling
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    # CNN 2 + Rectified Linear Unit + Subsampling
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    # CNN 3 + Rectified Linear Unit + Subsampling
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    # CNN 4 + Rectified Linear Unit
    model.add(Convolution2D(64,3,3, activation='relu'))
    # CNN 5 + Rectified Linear Unit
    model.add(Convolution2D(64,3,3, activation='relu'))
    # If keras 2+
    # model.add(Convolution2D(24,(5,5), strides=(2,2), activation='relu'))
    # model.add(Convolution2D(36,(5,5), strides=(2,2), activation='relu'))
    # model.add(Convolution2D(48,(5,5), strides=(2,2), activation='relu'))
    # model.add(Convolution2D(64,(3,3), activation='relu'))
    # model.add(Convolution2D(64,(3,3), activation='relu'))

    # Flattening the layer
    model.add(Flatten())
    # Adding dropout
    model.add(Dropout(0.5))
    # Fully Connected layer with 100 cells
    model.add(Dense(100))
    # Adding dropout
    model.add(Dropout(0.5))
    # Fully Connected layer with 50 cells
    model.add(Dense(50))
    # Fully Connected layer with 10 cells
    model.add(Dense(10))
    # Fully Connected layer with 1 cell which gives the regression output i.e the steering angle
    model.add(Dense(1))
    return model

	
# Reading images and merging the three arrays along with correction addition
DATA_DIR='data'
imagePaths, measurements = getImagePathsAndCorrectedMeasurements(DATA_DIR, 0.2)
print('Total Image Count: {}'.format( len(imagePaths)))

# Splitting data into training and validation sets
samples = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Total Trainig samples count: {}'.format(len(train_samples)))
print('Total Validation samples count: {}'.format(len(validation_samples)))

# Creating train and validation generators along with addition of flipped images.
training_data_generator = generator(train_samples, batch_size=32)
validation_data_generator = generator(validation_samples, batch_size=32)

# Creating Model
model = loadNvidiaModel()

# Compiling and training the model. NOTE: keras has a different syntax of fit_generator its twi versions of 1 and 2
model.compile(loss='mse', optimizer='adam')
batch_size = 64
epochs = 5

print("Found Keras Version 1")
print(keras.__version__)
history_object = model.fit_generator(training_data_generator, samples_per_epoch = len(train_samples) , validation_data=validation_data_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs, verbose=1)
# print("Found Keras Version 2")
# print(keras.__version__)
# steps_per_epoch = len(train_samples)/batch_size
# validation_steps = len(validation_samples)/batch_size
# history_object = model.fit_generator(training_data_generator, steps_per_epoch= steps_per_epoch , validation_data=validation_data_generator, validation_steps=validation_steps, epochs=epochs, verbose=1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()