# Multi-image classification with 3D CT scans using CNN

## 📂 Project Structure & Data Format
This project is designed as a generalizable framework for 3D CT scan classification tasks. While it is currently applied to lung cancer histopathological subtypes, it can be adapted to other CT-based medical imaging classification problems.

## 📁 Expected Directory Structure 
For this lung subtype example, CT scan data should be organized by **class label (subtype/diagnosis)**, with each patient containing their own DICOM image folder.

**Example Below:**

Class → Patient → slices

## Clone GitHub repo

```
# Bash

git clone https://github.com/Low-is/3D-CT-scans-image-classification.git
cd 3D-CT-scans-image-classification
```

## Create Python environment inside repo 
```
# Bash

python -m venv venv
```

## Activate Python environment
```
# Bash

source venv/Scripts/activate # For Git Bash, use this one

.\venv\Scripts\Activate # For WindowsPowerShell
```

## Install dependencies
```
# Bash

pip install -r requirements.txt
```

## Run training/train.py script
```
# Bash

python -m training.train.py
```

# Load trained model + evaluation data
```
# Python

import numpy as np
import json
from tensorflow import keras

model = keras.models.load_model("3dcnn_final.keras")

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
y_test_cat = np.load("y_test_cat.npy")

with open("history.json", "r") as f:
history = json.load(f)
```


# Visualizae CT scan slices (optional)
```
# Python

from dataset import build_dataset
from evaluation.plots import plot_volume_slices

x, y = build_dataset()  

plot_volume_slices(x[0])
```


# Evalulate model performance
```
# Python

from evaluation.metrics import evaluate_model

results = evaluate_model(model, x_test, y_test_cat)
print(results)
```


# Plot results
```
# Python

from evaluation.plots import (
plot_training_history,
plot_confusion_matrix
)

import numpy as np

# Training curves
plot_training_history(history)

# Predictions
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# Confusion matrix
plot_confusion_matrix(y_true, y_pred)
```


Convolutional neural networks (CNNs) is a type of deep learning model that is best for image processing. CNNs uses a system that looks at small patches of an image, finding patterns (like edges, textures, shapes, etc.) and then builds up a complex understanding from simple patterns, image classification. 

A typical CNN has 3 major building blocks:
1. Convultion layer:
     - Applies a filter (like a sliding window) over the image to extract features. These filters learn and detect patterns like: edges, corners, textures, and shapes.
     - Output: a new image-like array called a feature map.
2. Pooling layer:
     - Reduces the size of the feature maps while keeping the most important information.
     - The biggest values from each region along the image are retained.
3. Fully connected (Dense) layer:
     - After enough convolution + pooling layers, next you flatten the 2D feature maps into a 1D vector.
     - A fully connected layer is where each neuron receives input from all neurons in the previous layer. Flattening converts the multi-dimensional output of the convolutional and pooling layers into a one-dimensional vector, preparing it for input into a fully-connected (dense) layer. 

Lung cancer is hte 3rd most common cancer in the U.S. and is the leading cause of cancer-related deaths. It is a hetergenous disease categorized into two main kinds:
- Non-small cell lung cancer (NSCLC)
- Small cell lung cancer (SCLC)
NSCLC is the most prevalent type of lung cancer cases. Major histological types of NSCLC include: adenocarcinoma (ADC) and squamous cell carcinoma (SCC). SCLC grows more quickly and is harder to treat the NSCLC. It's often found as a relatively small lung tumor that has alredy spread to other parts of the body.

In clinical settings, histological classification is typically performed using manual examination of tissue sampels under a standard light microscope. While this method is considered reliable, biopsies can sometimes miss the full range of morhphological and phenotypic variations present in the tumor to due inter- and intra-tumor heterogeneity. Due to the complexity of lung cancer classification and the limitations in current practices, there is a need for innovative clinical assessment tools to, such as prediction models tht can help guide, speed up, or refine treatment decisions, espeically in early or inaccessible settings. 

Using publicly available CT scan images from the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/), 251,135 lung cancer images of 4 distinct tissue histopathological diagnosis (Adenocarcinoma, Small Cell Carcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma) across 436 studies and 355 subjects were used for multi-class image classification. The hypothesis is that using CNN, 4 distinct tissue histopathological cancers can be classified from 3D CT images.  


Citations:
- [Cancer Imaging Archives](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)
- [Cleveland Clinic](https://my.clevelandclinic.org/health/diseases/4375-lung-cancer)
- Chaunzwa, T.L. et al. *Deep learning classification of lung cancer histology using CT images.* Scientific Reports, **11**, 5471 (2021).
- Ilié, M. & Hofman, P. *Pros: Can tissue biopsy be replaced by liquid biopsy?* Translational Lung Cancer Research, **5**, 420–423 (2016).
- Zhao, B. et al. *Reproducibility of radiomics for deciphering tumor phenotype with imaging.* Scientific Reports, **6**, 23428 (2016).
