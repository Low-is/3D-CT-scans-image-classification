import yaml
import json
import sys
import os
import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# As long as Python knows the root folder '3D-CT-scans-image-classification/...', other scripts and their functions can be accessed
from models.cnn_3d import get_model
from data_pipeline.dataset import build_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def main():

    # -----------------------
    # LOAD CONFIG
    # -----------------------
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # -----------------------
    # LOAD DATA
    # -----------------------
    x, y = build_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y if config["split"]["stratify"] else None
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
    width, height, depth = config["data"]["image_shape"]
    
    model = get_model(width, height, depth, num_classes)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        config["training"]["learning_rate"],
        decay_steps=100000,
        decay_rate=0.96
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=config["training"]["loss"],
        metrics=["accuracy"]
    )

    # -----------------------
    # TRAIN
    # -----------------------
    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        class_weight=class_weights_dict
    )


    # -----------------------
    # CREATING DIRECTORIES
    # -----------------------
    model_dir = config["output"]["model_dir"]
    data_dir = config["output"]["data_dir"]
    log_dir = config["output"]["log_dir"]

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    

    # -----------------------
    # SAVE
    # -----------------------

    model.save(os.path.join(model_dir, "3dcnn_final.keras"))

    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    np.save(os.path.join(data_dir, "y_test_cat.npy"), y_test_cat)

    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(history.history, f)


if __name__ == "__main__":
    main()
