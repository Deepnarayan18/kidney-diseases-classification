import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Base model layers ko configure karna
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            # Starting ki layers ko freeze karo (general features like edges/textures)
            for layer in model.layers[:freeze_till]:
                layer.trainable = False
            # Aakhri ki layers ko unfreeze rakho (taaki ye kidney-specific tumor/cyst seekh sake)
            for layer in model.layers[freeze_till:]:
                layer.trainable = True

        # --- PRO CHANGES HERE ---
        # Flatten ki jagah GlobalAveragePooling2D use karo (better for medical spatial data)
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        
        # Ek intermediate Dense layer with ReLU to learn complex features
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # Dropout to prevent overfitting (50% neurons off randomly during training)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Final classification layer
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(x)
        # ------------------------

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Better Optimizer (Adam generally converges better for medical imaging than standard SGD)
        # Added Precision and Recall metrics
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        # --- FINE TUNING LOGIC UPDATE ---
        # Ab hum 'freeze_all=False' bhej rahe hain
        # 'freeze_till=15' ka matlab VGG16 ke total 19 layers me se, pehle 15 freeze rahenge
        # aur aakhri block (block5_conv1, conv2, conv3) train hoga.
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False, 
            freeze_till=15, 
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Format explicitly set taaki Keras 3 issues na aaye
        model.save(path)