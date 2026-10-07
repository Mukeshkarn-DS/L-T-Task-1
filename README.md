# L&T EduTech – Deep Learning Task Assignment

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Tasks](https://img.shields.io/badge/Tasks-1--8-success)](#task-overview)

## About

This repository contains my practical implementation work for the **L&T EduTech Deep Learning task assignment**. The work progresses from neural-network fundamentals to CNNs, transfer learning, object detection and segmentation, sequence modelling, representation learning, generative AI, and model optimization.

Each task is organized in a separate folder so that the implementation, notebooks, scripts, and generated results can be reviewed independently.

> **Academic Work:** This repository is maintained for learning, experimentation, evaluation, and academic demonstration.

---

## Task Overview

| Task | Topic | Major Concepts | Framework |
|---|---|---|---|
| **Task 1** | Neural Network Fundamentals & Experiments | Dense networks, activations, optimizers, batch-size experiments, hyperparameter tuning, error analysis | Python / NumPy |
| **Task 2** | Regularization & Bias–Variance Analysis | L1/L2, Dropout, Batch Normalization, model comparison | TensorFlow / Keras |
| **Task 3** | CNN Classification & Data Augmentation | CNN, CIFAR-10, augmentation, baseline vs augmented model | TensorFlow / Keras |
| **Task 4** | Advanced CNN & Transfer Learning | AlexNet-style CNN, transfer learning, evaluation and comparison | TensorFlow / Keras |
| **Task 5** | Object Detection & Segmentation | YOLO, Faster R-CNN, U-Net, mAP, IoU, Dice Score | PyTorch / TensorFlow |
| **Task 6** | Sequence Modeling | RNN, LSTM, sentiment classification, time-series prediction | TensorFlow / Keras |
| **Task 7** | Representation Learning | Autoencoder, VAE, reconstruction, latent-space interpolation | PyTorch |
| **Task 8** | Generative AI & Optimization | DCGAN, pruning, INT8 quantization, knowledge distillation | PyTorch |

---

# Detailed Task Documentation

## Task 1 — Neural Network Fundamentals & Experimental Analysis

**Folder:** L&T TASK 1/

The first task focuses on implementing and analysing fundamental neural-network components.

### Work covered

- Dense neural-network architecture
- ReLU and Softmax activation functions
- Forward and backward propagation
- Cross-entropy loss
- SGD, Momentum, RMSProp and Adam
- Batch-size experiments
- Single vs multi-configuration experiments
- Hyperparameter tuning
- Training-curve analysis
- Confusion matrix
- Misclassified-image analysis

### Implementation

The task contains custom neural-network components in the **nn/** folder and supporting scripts including:

- train.py
- compare_activations.py
- compare_batch_sizes.py
- compare_optimizers.py
- tune_hyperparameters.py
- visualize_errors.py

### Generated results

The folder includes activation, optimizer and batch-mode comparisons, training curves, confusion matrix, misclassified images, and hyperparameter results.

---

## Task 2 — Regularization and Bias–Variance Analysis

**Folder:** L&T TASK 2/

Task 2 studies methods used to reduce overfitting and improve generalization.

### Work covered

- CNN model construction
- L1 regularization
- L2 regularization
- Dropout
- Batch Normalization
- Training and validation analysis
- Bias–variance analysis
- Model-performance comparison

### Reference datasets

The notebook supports:

- MNIST
- Fashion-MNIST
- CIFAR-10

The dataset is selected according to the experiment and task objective.

---

## Task 3 — CNN Image Classification and Data Augmentation

**Folder:** L&T TASK 3/

Task 3 implements CNN-based image classification using **CIFAR-10**.

### Work covered

- CIFAR-10 preprocessing
- Baseline CNN
- Convolution and pooling
- Batch Normalization
- Dropout
- Data augmentation
- Augmented CNN
- Training-history analysis
- Classification metrics
- Confusion matrix
- Baseline vs augmented model comparison

**Notebook:** L&T_TASK3.ipynb

### Objective

To demonstrate how CNNs learn visual features and how data augmentation can improve robustness and generalization.

---

## Task 4 — Advanced CNN, AlexNet and Transfer Learning

**Folder:** L&T TASK 4/

Task 4 focuses on advanced image-classification approaches and transfer learning.

### Included notebooks

- 01_CIFAR_10_Class_Classifier_with_AlexNet_architecture.ipynb
- Transfer_learning_cat_vs_dog.ipynb
- L&T_Task_4_Evaluation_and_Comparison.ipynb

### Work covered

- AlexNet-style CNN architecture
- CIFAR-10 classification
- Transfer learning
- Cat-vs-dog classification
- Test accuracy and loss
- Precision, recall and F1-score
- Confusion matrix
- Comparative model analysis

The dedicated evaluation notebook provides a structured evaluation and comparison of the trained models.

---

## Task 5 — Object Detection and Image Segmentation

**Folder:** L&T TASK 5/

Task 5 extends computer vision from image classification to object localization and pixel-level segmentation.

### Practical components

**YOLO Object Detection**
- Construction/PPE object detection
- Prediction visualization
- mAP@50
- mAP@50:95

**Faster R-CNN**
- Road-sign object detection workflow
- Detection evaluation

**U-Net**
- Semantic segmentation
- Oxford-IIIT Pet dataset
- Pixel-level prediction

### Evaluation metrics

- mAP
- IoU
- Dice Score
- Prediction visualizations

**Notebook:** L&T_TASK_5 (1).ipynb

> GPU execution is recommended for the computationally intensive detection and segmentation experiments.

---

## Task 6 — Sequence Modeling using RNN and LSTM

**Folder:** L&T TASK 6/

Task 6 introduces recurrent neural networks for sequential and time-series data.

### Work covered

- Sequence preprocessing
- Embedding
- Simple RNN
- LSTM
- Dropout
- Early stopping
- Classification evaluation
- Time-series modelling
- RNN vs LSTM comparison

### Reference datasets

- **IMDB Movie Reviews** — sentiment classification
- **Airline Passenger Dataset** — time-series prediction

**Notebook:** L&T_TASK_6.ipynb

### Objective

To understand how recurrent architectures process sequential information and why LSTM networks are effective for longer-term dependencies.

---

## Task 7 — Autoencoders and Variational Autoencoders

**Folder:** L&T TASK 7/

Task 7 focuses on unsupervised representation learning.

### Dataset

**Fashion-MNIST**

### Work covered

- Encoder and decoder architecture
- Latent representation
- Image reconstruction
- Autoencoder (AE)
- Variational Autoencoder (VAE)
- Latent-space analysis
- AE vs VAE comparison
- Latent-space interpolation
- Reconstruction-quality analysis

**Notebook:** L&T_TASK_7.ipynb

### Objective

To demonstrate how neural networks can learn compact representations and how VAEs create structured latent spaces for generative applications.

---

## Task 8 — Generative AI and Deep Learning Model Optimization

**Folder:** L&T TASK 8/

Task 8 combines generative modelling with practical model-optimization techniques.

### Part A — DCGAN

A **Deep Convolutional Generative Adversarial Network** is implemented on MNIST.

Components include:

- Generator
- Discriminator
- Adversarial training loop
- Generator/discriminator loss tracking
- Generated-image visualization
- Generator weight saving

### Part B — Model Optimization

The notebook covers:

**Pruning** — reducing unnecessary model parameters.

**Dynamic INT8 Quantization** — reducing numerical precision to improve efficiency.

**Knowledge Distillation** — transferring knowledge from a larger teacher model to a smaller student model.

### Comparison

The optimization methods are compared using measures such as:

- Parameter count
- Model size
- Accuracy
- Efficiency
- Compression impact

**Notebook:** L&t_Task_8.ipynb

---

# Repository Structure

~~~text
L_T_-Task-_Assignment/
│
├── L&T TASK 1/
│   ├── nn/
│   ├── train.py
│   ├── compare_activations.py
│   ├── compare_batch_sizes.py
│   ├── compare_optimizers.py
│   ├── tune_hyperparameters.py
│   ├── visualize_errors.py
│   └── generated results
│
├── L&T TASK 2/
│   └── L&T_TASK_2.ipynb
│
├── L&T TASK 3/
│   └── L&T_TASK3.ipynb
│
├── L&T TASK 4/
│   ├── 01_CIFAR_10_Class_Classifier_with_AlexNet_architecture.ipynb
│   ├── Transfer_learning_cat_vs_dog.ipynb
│   └── L&T_Task_4_Evaluation_and_Comparison.ipynb
│
├── L&T TASK 5/
│   └── L&T_TASK_5 (1).ipynb
│
├── L&T TASK 6/
│   └── L&T_TASK_6.ipynb
│
├── L&T TASK 7/
│   └── L&T_TASK_7.ipynb
│
└── L&T TASK 8/
    └── L&t_Task_8.ipynb
~~~

---

# Technology Stack

### Programming
- Python
- Jupyter Notebook
- Google Colab

### Deep Learning
- TensorFlow
- Keras
- PyTorch
- Torchvision
- Ultralytics YOLO

### Data Science and Visualization
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

# Datasets

| Dataset | Main Usage |
|---|---|
| **MNIST** | Neural networks and generative modelling |
| **Fashion-MNIST** | Regularization and AE/VAE experiments |
| **CIFAR-10** | CNN and AlexNet classification |
| **IMDB Movie Reviews** | RNN/LSTM sentiment classification |
| **Airline Passenger Dataset** | Time-series sequence modelling |
| **Oxford-IIIT Pet** | Semantic segmentation |
| **Construction PPE Dataset** | YOLO object detection |
| **Road-Sign Dataset** | Faster R-CNN object detection |

> The L&T task guidelines provide multiple reference datasets for several tasks. This repository uses the dataset or datasets that best match each practical objective; every reference dataset does not need to be used in every task.

---

# Learning Progression

The eight tasks provide a structured progression:

1. **Neural-network fundamentals**
2. **Regularization and generalization**
3. **Convolutional image classification**
4. **Advanced CNNs and transfer learning**
5. **Object detection and semantic segmentation**
6. **Sequential modelling with RNN/LSTM**
7. **Representation learning with AE/VAE**
8. **Generative AI and model optimization**

This progression covers both **model development** and **model evaluation**, including accuracy, loss, F1-score, confusion matrices, mAP, IoU, Dice Score, reconstruction quality, parameter count, model size, and optimization efficiency.

---

# How to Run

## Google Colab

1. Open the required notebook.
2. Open it in Google Colab.
3. Select a GPU runtime for computationally intensive tasks.
4. Run the notebook cells sequentially.
5. Review the generated metrics, visualizations, and model outputs.

## Local Jupyter

Install the core dependencies:

~~~bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torch torchvision
~~~

Then launch Jupyter:

~~~bash
jupyter notebook
~~~

Task 5 may require additional packages such as Ultralytics and TorchMetrics. The required installation commands are included in the relevant notebook.

---

# Reproducibility

Where applicable, experiments use fixed random seeds to improve reproducibility.

Typical configuration:

~~~python
SEED = 42
~~~

GPU acceleration is recommended for Tasks 4–8.

---

# Evaluation

Depending on the task, the repository includes:

- Training and validation curves
- Accuracy and loss
- Precision
- Recall
- F1-score
- Confusion matrix
- Misclassified samples
- mAP@50
- mAP@50:95
- IoU
- Dice Score
- Reconstruction analysis
- Latent-space analysis
- Parameter-count comparison
- Model-size comparison
- Optimization and compression analysis

---

# Author

**Mukesh Kumar**  
B.Sc. Data Science

GitHub: [@Mukeshkarn-DS](https://github.com/Mukeshkarn-DS)

## Repository

[L&T EduTech – Task Assignment](https://github.com/Mukeshkarn-DS/L_T_-Task-_Assignment)

---

# Academic Note

This repository represents practical implementation and experimentation performed as part of the **L&T EduTech Deep Learning task assignment**. The notebooks are organized to demonstrate concepts, implementation workflows, evaluation methods, and observations for Tasks 1–8.

---

## License

This repository is maintained for educational and academic purposes.
