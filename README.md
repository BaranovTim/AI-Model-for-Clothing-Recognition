# FashionMNIST Classification with PyTorch #
### Description ###

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify images from the FashionMNIST dataset. The system trains a model capable of recognizing various clothing items (such as shirts, shoes, and bags) and evaluates its performance using precision, recall, F1-score, and visualization metrics.

The project uses a custom-designed CNN architecture built from scratch, featuring multiple convolutional layers, activation functions, and pooling layers. It also includes a custom MinPool2d module, demonstrating experimentation with non-standard pooling operations.
Extensive evaluation functions are implemented to analyze model performance and generate precision–recall curves, confusion matrices, and F1 vs confidence graphs.

### Key Features ###
1. <ins> Model Architecture </ins>

Built from scratch using PyTorch (torch.nn.Module).

Two convolutional stacks with Conv2d, ReLU, and AvgPool2d layers.

A fully connected output layer classifying 10 clothing categories.

Includes a custom MinPool2d class that performs inverse pooling (the minimum operation).

2. <ins> Dataset </ins>

Utilizes FashionMNIST, a standard dataset of 28×28 grayscale clothing images.

Automatically downloaded using torchvision.datasets.FashionMNIST.

Divided into training and testing sets with batch loading via DataLoader.

3. <ins> Training and Testing </ins>

Training loop supports backpropagation and gradient descent using SGD.

Real-time accuracy and loss tracking per epoch.

Test evaluation computes average loss and accuracy on unseen data.

4. <ins> Evaluation Metrics </ins>

Calculates Precision, Recall, and F1-score for all classes.

Generates detailed classification reports using sklearn.metrics.

Plots include:

Precision–Recall curves per class and macro average.

F1 vs confidence threshold visualization.

Precision & Recall vs confidence.

Confusion Matrix displaying classification performance.

5. <ins> Visualization </ins>

Displays random test predictions with color-coded results:

Green = Correct predictions

Red = Incorrect predictions

Uses Matplotlib to show prediction grids and performance curves.

6. <ins> Device Optimization </ins>

Automatically detects and utilizes GPU (CUDA) if available.

Efficient handling of tensors between CPU and GPU for speed.

### Requirements ###

Install the required dependencies:

pip install torch torchvision matplotlib scikit-learn numpy

### Usage ###

Clone the repository or copy the project files.

Train and test the model:

python main.py


After training, visualizations of metrics and confusion matrices will appear automatically.

### Output Examples ###

Training Accuracy and Loss printed for each epoch.

Precision–Recall Curves for all 10 classes.

F1-score vs Confidence plots to evaluate threshold sensitivity.

Confusion Matrix summarizing predictions vs. true labels.

Sample Predictions Grid showing predicted and actual clothing types.
