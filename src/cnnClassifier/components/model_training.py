import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # --- PRO CHANGE 1: Optimized Medical Data Augmentation ---
        # CT scans ko 40 degrees rotate/shear karna anatomy ko distort kar deta hai. 
        # Isliye values kam aur natural rakhi hain.
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,          # Reduced from 40 (CT scans mein up/down fix hota hai)
                horizontal_flip=True,       # Left/Right kidney mirror ho sakti hai
                vertical_flip=False,        # CT scans ulte nahi hote
                width_shift_range=0.1,      # Reduced from 0.2
                height_shift_range=0.1,     # Reduced from 0.2
                shear_range=0.1,            # Reduced from 0.2
                zoom_range=0.1,             # Reduced from 0.2
                fill_mode="nearest",
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        
        # --- PRO CHANGE 2: Calculate Class Weights Dynamically ---
        # Ye imbalance ko handle karega aur minor classes (Tumor/Cyst) ko zyada priority dega
        classes = self.train_generator.classes
        class_weights_arr = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(classes),
            y=classes
        )
        self.class_weights = dict(enumerate(class_weights_arr))

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # --- PRO CHANGE 3: Advanced Callbacks ---
        # Training ko best epoch par rokne aur LR ko adjust karne ke liye
        
        # 1. Model Checkpoint: Hamesha best validation recall wala model save karo
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.trained_model_path),
            monitor="val_recall",  # Humesha check karega ki tumor kitne miss hue
            mode="max",
            save_best_only=True,
            verbose=1
        )

        # 2. Early Stopping: Agar 7 epochs tak recall nahi badhta to training rok do
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )

        # 3. Reduce LR on Plateau: Agar validation loss ruk jaye to Learning Rate aur kam kardo
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )

        callbacks_list = [checkpoint, early_stopping, reduce_lr]

        # Training ke andar callbacks aur class_weights pass kiye gaye hain
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callbacks_list,
            class_weight=self.class_weights
        )
        
        # Checkpoint khud best model save kar dega, par in case last epoch model save karna ho
        # humne save_model ko pehle hi checkpoint me handle kar liya hai.