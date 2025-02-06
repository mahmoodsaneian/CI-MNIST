# CI-MNIST

## Overview  
**CI-MNIST** is a machine learning project aimed at classifying handwritten digits from the **MNIST** dataset using **Convolutional Neural Networks (CNNs)** or other deep learning techniques. This project demonstrates how to build, train, and evaluate a neural network on the **MNIST dataset**, one of the most famous datasets in the machine learning community.

## Features  
- **Digit Classification**: Classifies 28x28 pixel grayscale images of handwritten digits (0-9).  
- **Neural Network Architecture**: Uses **Convolutional Neural Networks (CNNs)** for feature extraction and classification.  
- **Model Evaluation**: Evaluates the performance of the model using accuracy, loss, and confusion matrices.  
- **Data Preprocessing**: Prepares the MNIST dataset for training, including normalization and reshaping.  

## Technologies Used  
- **Python 3.x**  
- **TensorFlow** or **Keras** for deep learning  
- **NumPy** and **Pandas** for data manipulation  
- **Matplotlib** for visualizations (accuracy/loss graphs)  

## Requirements  
- Python 3.x  
- TensorFlow or Keras  
- NumPy, Pandas, Matplotlib  

You can install the required dependencies using `pip`:

```sh
pip install -r requirements.txt
```

## Setup  

### Clone the Repository  
```sh
git clone https://github.com/mahmoodsaneian/CI-MNIST.git
cd CI-MNIST
```

### How to Run  
1. Install dependencies:  
   ```sh
   pip install -r requirements.txt
   ```
2. Train the Model:  
   ```sh
   python train.py
   ```
   This script will load the MNIST dataset, preprocess the data, train the CNN model, and save the trained model to disk.
3. Test the Model:  
   ```sh
   python test.py
   ```
   This script will load the saved model, evaluate its performance on the test set, and display the results.

### Input Format  
- The input is a 28x28 pixel grayscale image representing a handwritten digit from the MNIST dataset.

### Output  
- The output is the predicted digit (0-9) for the given input image.  
- The model's performance is evaluated using accuracy and loss metrics.

## Model Details  
- The **Convolutional Neural Network (CNN)** architecture typically used for MNIST includes:  
  - **Convolutional Layers**: To learn features from the input images.  
  - **MaxPooling Layers**: To reduce the spatial dimensions of the feature maps.  
  - **Fully Connected Layers**: To classify the image based on the learned features.  
- The model is trained using **Categorical Crossentropy** as the loss function and **Adam optimizer** for gradient descent.

## Project Structure  
```
CI-MNIST/
│── src/
│   ├── train.py (Training script)
│   ├── test.py (Testing and evaluation script)
│── README.md
│── requirements.txt (Dependencies)
│── .gitignore
```  

## Contributing  
Feel free to fork the repository, open issues, or submit pull requests with improvements, bug fixes, or new features.  

## License  
This project is licensed under the MIT License.

## Repository  
[GitHub: CI-MNIST](https://github.com/mahmoodsaneian/CI-MNIST)
