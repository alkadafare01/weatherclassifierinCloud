from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
#from tensorflow_core.python.client import session
import pathlib

session = 'simpleNASNet'
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu',kernel_initializer='he_uniform', padding='same'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))
# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu',kernel_initializer='he_uniform', padding='same'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))
#Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Dropout(0.2))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 5, activation = 'softmax'))

# Compiling the CNN
classifier.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
logfile = session + '-train' + '.log'
csv_logger = CSVLogger(logfile, append=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
best_model_filename=session+'-weights.{epoch:02d}-{val_loss:.2f}.h5'
best_model = ModelCheckpoint(best_model_filename, monitor='val_acc', verbose=1, save_best_only=True)
# this is the augmentation configuration we will use for training
##classifier.fit_generator(
  #   generator=training_set,
   #  epochs=10,
    # verbose=1,
     #validation_data=test_set,
     #callbacks=[best_model, csv_logger, early_stopping])##
model = classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 32)

classifier.save("model.h5")
print("Saved model to disk")

# Part 3 - Making new predictionspyth

