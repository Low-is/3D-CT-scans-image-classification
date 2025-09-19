# Multi-image classification with 3D CT scans using CNN

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


## Performance Metrics
The model achieved:
- Training Accuracy: 75%
- Training Loss: 0.90
- Validation Accuracy: 31% (still needs improvement)
- Validation Loss: 1.99 (still needs improvement)

[!model](3D_model.png)


Citations:
- [Cancer Imaging Archives](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/)
- [Cleveland Clinic](https://my.clevelandclinic.org/health/diseases/4375-lung-cancer)
- Chaunzwa, T.L. et al. *Deep learning classification of lung cancer histology using CT images.* Scientific Reports, **11**, 5471 (2021).
- Ilié, M. & Hofman, P. *Pros: Can tissue biopsy be replaced by liquid biopsy?* Translational Lung Cancer Research, **5**, 420–423 (2016).
- Zhao, B. et al. *Reproducibility of radiomics for deciphering tumor phenotype with imaging.* Scientific Reports, **6**, 23428 (2016).
