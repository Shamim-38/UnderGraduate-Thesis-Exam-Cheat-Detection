"""
Author: Shamim
"""

import keras
import numpy as np 
import os

# project modules
from .. import config
from . import my_model, make_dataset_3dcd


# loading data
cheat_type_list = os.listdir(config.pose_3dcd_path())
cheat_type_list = sorted(cheat_type_list)
X_train, y_train = make_dataset_3dcd.get_keypoints_for_all_cheat(cheat_type_list)

print("train data shape: ", X_train.shape)
print("train data label: ", y_train.shape)


# laoding model
model = my_model.get_model()

# compile
model.compile(keras.optimizers.Adam(config.lr), 
            keras.losses.categorical_crossentropy,
            metrics=['accuracy'])


# checkpoins
model_cp = my_model.save_model_checkpoint()
early_stopping = my_model.set_early_stopping()


# for training model
model.fit(X_train, y_train, 
        batch_size = config.batch_size, 
        epochs =config.nb_epochs,  
        verbose = 2, 
        shuffle = True, 
        callbacks = [early_stopping, model_cp], 
        validation_split = 0.2)