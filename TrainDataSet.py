import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Parameters
SIZE = 250
INPUT_SHAPE = (SIZE, SIZE, 3)
batch_size = 16
num_classes = 10
epochs = 50

# Directories
train_dir = 'image_classes'
test_dir = 'test'

# Data preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(SIZE, SIZE),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model building
model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(32))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=[checkpoint, early_stop]
    )

    # Save the final model to .keras file
    model.save('model.keras')

    # Save the model architecture to .json file
    model_json = model.to_json()
    with open('model.json', 'w') as json_file, open('classes.json', 'w') as json_classes:
        json_file.write(model_json)
        json_classes.write(json.dumps(train_generator.class_indices))

    print("Model saved to model.keras and model.json")
except Exception as e:
    print(f"Error occurred during training: {e}")

# Load the best model and evaluate it on the test set
try:
    best_model = tf.keras.models.load_model('best_model.keras')
    test_loss, test_accuracy = best_model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
except Exception as e:
    print(f"Error occurred during testing: {e}")
