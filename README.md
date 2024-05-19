# Heart Disease Classification Using Particle Swarm Optimization (PSO)

This project focuses on the classification of a heart disease dataset using a neural network optimized with Particle Swarm Optimization (PSO). The performance of the PSO-optimized neural network is evaluated and compared to other optimization techniques.

## Dataset

The dataset used for this project is the [Heart Disease Dataset](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset) from Kaggle. The dataset contains the following attributes:

1. Age
2. Sex
3. Chest pain type
4. Resting blood pressure
5. Serum cholesterol
6. Fasting blood sugar
7. Resting electrocardiogram results
8. Maximum heart rate achieved
9. Exercise induced angina
10. Oldpeak = ST depression induced by exercise
11. The slope of the peak exercise ST segment
12. Target (0: No heart disease, 1: Heart disease)

## Project Structure

The project consists of the following files:

- `Code.ipynb`: Jupyter Notebook containing the implementation of the PSO optimization for neural network weights.
- `README.md`: Project documentation file.

## Installation

To run this project, you need to have the following libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- keras
- tensorflow
- pyswarms

You can install the required libraries using pip:

```bash
!pip install pandas numpy scikit-learn matplotlib keras tensorflow pyswarms --quiet
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/khaireddine-arbouch/Heart-Disease-Classification-Using-Particle-Swarm-Optimization-PSO-.git
cd heart-disease-classification-PSO
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook Code.ipynb
```
3. Run the notebook to train the neural network using PSO and generate the comparison table.

## Particle Swarm Optimization (PSO)
PSO is implemented using the PySwarms library to optimize the weights of a one hidden layer neural network. The PSO algorithm updates the weights of the neural network based on the heart disease dataset.

## Steps for PSO Implementation
- Load and preprocess dataset: The dataset is loaded, preprocessed, and split into training and testing sets.
- Define neural network model: A function create_model is defined to create a neural network with weights passed as an argument.
- Fitness function: A fitness function is defined that creates a model with given weights, trains it, and returns the negative accuracy.
- Calculate dimensions: Correctly calculates the total number of weights and biases.
- PSO optimization: Sets up and performs the PSO optimization.
- Train and evaluate the final model: The final model is trained with the optimized weights and its accuracy is evaluated on the test set.

## Results
The PSO optimization achieved an accuracy of 85.29%, demonstrating its effectiveness in optimizing neural network weights.

## Summary
The implementation of Particle Swarm Optimization (PSO) on the heart disease dataset yielded significant results. PSO achieved an accuracy of 85.29%, demonstrating its effectiveness in optimizing neural network weights. This highlights the potential of using advanced optimization techniques to improve the performance of neural network models in medical data classification.

## Acknowledgments
- The Heart Disease Dataset is provided by Kaggle.
- The PySwarms library for Particle Swarm Optimization implementation.
