# <span style="font-size:larger;">Chest X-Ray Image Classification</span>

This program performs image classification on chest X-ray images to detect pneumonia. It utilizes TensorFlow and a ResNet50V2-based deep learning model for accurate classification.

## <span style="font-size:larger;">Dataset</span>

The dataset used for training, validation, and testing is the Chest X-Ray dataset from Kaggle. The dataset, provided by Paul Mooney, contains X-ray images categorized into two classes: "NORMAL" and "PNEUMONIA". The dataset can be found [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## <span style="font-size:larger;">Technologies and Libraries Used</span>

The following technologies and libraries are utilized in this project:

- TensorFlow: The program leverages TensorFlow, an open-source machine learning framework, for building and training the deep learning model.

- NumPy: The NumPy library is used for numerical computations and array manipulation.

- Matplotlib: The Matplotlib library is employed for plotting and visualizing the images.

- pathlib: The pathlib module is used for handling file paths in a platform-independent way.

## <span style="font-size:larger;">Usage</span>

To use this program, follow these steps:

1. Ensure that Python and the required libraries are installed.

2. Clone this repository to your local machine.

3. Download the Chest X-Ray dataset from Kaggle ([link](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)) and place it in the appropriate folders: "chest_xray/train" for training images, "chest_xray/test" for validation images, and "chest_xray/val" for testing images.

4. Run the provided code, preferably in a Python environment.

5. The program loads the dataset and preprocesses the images, including resizing and data augmentation.

6. It builds a deep learning model based on the ResNet50V2 architecture.

7. The model is trained on the training dataset and evaluated on the validation dataset.

8. The program outputs the training and validation metrics, such as accuracy, precision, recall, and loss.

9. The best weights of the model are saved as "best_weights.h5".

10. The model is saved as "model.h5" for future use.

11. Finally, the program loads the saved model and evaluates it on the test dataset.

## <span style="font-size:larger;">Results and Analysis</span>

The program generates predictions for pneumonia detection based on the provided chest X-ray images. It evaluates the performance of the trained model using metrics such as accuracy, precision, recall, and loss. Additionally, the program provides visualizations to compare the actual and predicted labels, allowing for further analysis of the model's performance.

