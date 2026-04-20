import keras
from keras import layers


def get_model(config, width=128, height=128, depth=64):
    """
    3D CNN for volumetric classification.
    Input shape: (width, height, depth, 1)
    """

    # =========================
    # EXTRACT CONFIG VALUES
    # =========================
    filters = config["model"]["filters"]
    kernel_size = config["model"]["kernel_size"]
    dropout = config["model"]["dropout"]
    dense_units = config["model"]["dense_units"]
    num_classes = config["data"]["num_classes"]
    
    # =========================
    # INPUT
    # =========================
    inputs = keras.Input((width, height, depth, 1))

    # Block 1
    x = layers.Conv3D(filters[0], kernel_size, activation="relu")(inputs)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)

    # Block 2
    x = layers.Conv3D(filters[1], kernel_size, activation="relu")(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)

    # Block 3
    x = layers.Conv3D(filters[2], kernel_size, activation="relu")(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)

    # Block 4
    x = layers.Conv3D(filters[3], kernel_size, activation="relu")(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.BatchNormalization()(x)

    # Head
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")

    return model
