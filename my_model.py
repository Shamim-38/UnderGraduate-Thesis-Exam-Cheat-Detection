"""
Author: md Shamim
File: Network architecture
"""

import keras, os
from keras.layers import (Dense, Activation, 
            Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU)
from keras.models import Sequential, load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping


# project modules
from .. import config

model_checkpoint_dir = os.path.join(config.checkpoint_path(), "chat_baseline.h5")
saved_model_dir = os.path.join(config.output_path(), "cheat_baseline.h5")



# defining CNN model
def get_model():
    model = Sequential()

    model.add(Conv2D(16, (2, 4), strides=(2,1), padding = "valid", 
                kernel_initializer='he_uniform',
                input_shape = config.shape))
    model.add(LeakyReLU(alpha = 0.1))

    model.add(Conv2D(32, (2, 4), padding = "valid", kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha = 0.1))

    model.add(MaxPooling2D(pool_size=(1,4), strides=(2, 1), padding="same"))

    model.add(Conv2D(64, (2, 4), strides=(2, 1), padding = "valid", 
                        kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha = 0.1))

    model.add(Conv2D(128, (2, 4), strides=(1, 1), padding = "valid", 
                        kernel_initializer='he_uniform'))
    model.add(LeakyReLU(alpha = 0.1))

    model.add(Flatten())
    model.add(Dense(config.nb_classes, activation="softmax"))
    
    return model



def read_model():
    model = load_model(saved_model_dir)
    return model



def save_model_checkpoint():
    return ModelCheckpoint(model_checkpoint_dir, 
                            monitor='val_loss', 
                            verbose = 2, 
                            save_best_only = True, 
                            save_weights_only = False, 
                            mode='auto', 
                            period = 1)


def set_early_stopping():
    return EarlyStopping(monitor='val_loss', 
                        patience = 15, 
                        verbose=2, 
                        mode='auto')


if __name__ == "__main__":
    m = get_model()
    m.summary()