Gradient Descent for Neural Network (Logistic Regression)
Project Overview

This project demonstrates the implementation of gradient descent for a neural network (logistic regression) to predict whether a person would buy life insurance based on their age and affordability.

This is a binary classification problem, meaning there are only two possible outcomes:

1 → Person buys insurance

0 → Person does not buy insurance

The project uses TensorFlow 2.0, Keras, Python, and supporting libraries such as NumPy, pandas, scikit-learn, and Matplotlib.

Dataset

The dataset insurance_data.csv contains three key columns: age, affordability (a binary indicator of whether a person can afford insurance), and bought_insurance (the target variable, 1 if bought insurance and 0 if not).

Example data points include ages like 22, 25, or 47, affordability as 0 or 1, and bought_insurance as 0 or 1 depending on whether the person purchased insurance.

Workflow

Data Loading
The dataset is loaded and inspected to understand its structure.

Data Preprocessing
The dataset is split into training and test sets. Features, especially age, are scaled to normalize the inputs for better neural network performance.

Model Building
A simple neural network with a single Dense layer is built using a sigmoid activation function for binary classification.

Training
The model is trained using gradient descent with binary cross-entropy loss. Early stopping is used to prevent overfitting and halt training when improvements plateau.

Evaluation
The model is evaluated on the test set using accuracy and loss metrics. Learned weights and bias are inspected to understand the model's behavior.

Prediction
Predictions are made for new data points using both the trained model and a manual calculation with the sigmoid function for verification.

Outcomes

The model learns weights and bias that define the relationship between age, affordability, and insurance purchase.

On the test set, the model achieves a loss of approximately 0.463 and an accuracy of around 85%.

For new data points, the model predicts probabilities, which can be converted into classes (0 or 1) to determine whether a person is likely to buy insurance.

Manual predictions using the learned weights and sigmoid function closely match model predictions.

Key Concepts

Gradient Descent: Optimization algorithm that minimizes the loss function by updating model weights iteratively.

Binary Cross-Entropy Loss: Measures how well the predicted probabilities match the true binary labels.

Sigmoid Activation: Converts weighted sum of inputs into a probability between 0 and 1.

Early Stopping: Prevents overfitting by stopping training once improvements plateau.

Tools & Libraries

This project uses Python for implementation, TensorFlow 2.0 and Keras for the neural network, NumPy for numerical computations, pandas for data manipulation, scikit-learn for preprocessing and splitting data, and Matplotlib for optional visualization.

Project Structure

The project typically contains the following files:

README.md

insurance_data.csv (the dataset)

gradient_descent_insurance.py (Python script for training, evaluation, and prediction)

References

TensorFlow Keras Documentation

Gradient Descent Concept on Wikipedia
