#Required Imports
import numpy as np
import tensorflow as tf  # For tf.data
import matplotlib.pyplot as plt
import keras
from keras import layers
from image_model.Preprocessing import preprocess_image

def build_model(num_classes):
    inputs = layers.Input(shape=(224, 224, 3))
    
    model = keras.applications.EfficientNetV2B1(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
    include_preprocessing=True,
    name="efficientnetv2-b1",
    )
    
    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="Enetv2b1")
    return model

def predict_from_frames(frames):
    model = build_model(num_classes=3)
    model.load_weights("./image_model/bestweights.h5.keras")
    for i in range(len(frames)):    
        frames[i] = preprocess_image(frames[i])
    return model.predict(np.array(frames))

