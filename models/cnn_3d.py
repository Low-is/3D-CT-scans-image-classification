import keras
from keras import layers


def get_model(width=128, height=128, depth=64, num_classes=4):
    """
    3D CNN for volumetric classification.
    Input shape: (width, height, depth, 1)
    """

    inputs = keras.Input((width, height, depth, 1))

    # Block 1
    x = layers.Conv3D(64, 3, activation="relu")(inputs)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)

    # Block 2
    x = layers.Conv3D(64, 3, activation="relu")(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)

    # Block 3
    x = layers.Conv3D(128, 3, activation="relu")(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)

    # Block 4
    x = layers.Conv3D(256, 3, activation="relu")(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)

    # Head
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")

    return model
