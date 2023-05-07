# Chest X-ray Pneumonia Detection
This project aims to build a binary classifier to detect whether a chest x-ray is normal or contains pneumonia using a convolutional neural network (CNN).

# Dataset
The dataset used in this project is the Chest X-ray Images (Pneumonia) from Kaggle, which contains a total of 5,856 chest x-ray images in JPEG format, with a total size of 1.2GB. The images are classified into two categories: "Pneumonia" and "Normal".

The dataset is split into three subsets: train, test, and validation, with a 80:10:10 split ratio.

# Preprocessing
The images are resized to 256x256 and converted to grayscale to reduce processing time. The images are then loaded using the tf.keras.utils.image_dataset_from_directory() function and passed to the CNN.

# Model
The CNN used in this project consists of two convolutional layers with a ReLU activation function and a max pooling layer, followed by a flatten layer, two dense layers with a ReLU activation function, and a sigmoid output layer. The model is compiled using binary cross-entropy loss and accuracy metrics.

# Training
The model is trained for 20 epochs on the training dataset, with a batch size of 32. Early stopping is implemented with a patience of 3 epochs to prevent overfitting. The validation dataset is used to evaluate the model's performance during training.

# Results
The model achieves a validation accuracy of 86.6% and a validation loss of 0.35. The training accuracy and loss plots are shown below:

Training Accuracy vs Validation Accuracy

Training Loss vs Validation Loss

![image](https://user-images.githubusercontent.com/109113298/236652794-0d51fb05-4e62-4952-8cb0-ded3e6ecc423.png)

# Prediction
The model is used to predict whether an x-ray is normal or contains pneumonia on the test dataset. The prediction results are shown below:

# Prediction Results

The model achieves an accuracy of 87.8% on the test dataset.

# Conclusion
The CNN model achieves a high accuracy in detecting pneumonia from chest x-ray images. This project demonstrates the potential of deep learning in medical imaging analysis and could be further improved with more data and fine-tuning of hyperparameters.
