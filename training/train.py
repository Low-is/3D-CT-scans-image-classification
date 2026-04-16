import json
import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# As long as Python knows the root folder '3D-CT-scans-image-classification/...', other scripts and their functions can be accessed
from models.cnn_3d import get_model
from dataset import build_dataset



def main():

    # -----------------------
    # LOAD DATA
    # -----------------------
    x, y = build_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    num_classes = y_train_cat.shape[1]

    # -----------------------
    # CLASS WEIGHTS
    # -----------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    # -----------------------
    # MODEL
    # -----------------------
    model = get_model(128, 128, 64, num_classes)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        1e-4, decay_steps=100000, decay_rate=0.96
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -----------------------
    # TRAIN
    # -----------------------
    history = model.fit(
        x_train, y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=50,
        batch_size=2,
        class_weight=class_weights_dict
    )

    # -----------------------
    # SAVE
    # -----------------------
    model.save("3dcnn_final.keras")
    np.save("x_train.npy", x_train)
    np.save("x_test.npy", x_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("y_test_cat.npy", y_test_cat)
    
    with open("history.json", "w") as f:
        json.dump(history.history, f)


if __name__ == "__main__":
    main()
