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
     - A fully connected layer is where each neuron receives input from all neurons in the previous layer. Flattening converts the multi-dimensional output of the convolutional and pooling layers into a one-dimensional vector, preparing it for inout into a fully-connected (dense) layer. 
