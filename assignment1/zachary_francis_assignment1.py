#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_diabetes_data():
    diabetes_bunch = load_diabetes()
    feature_array = np.array(diabetes_bunch['data'])
    target_array = np.array(diabetes_bunch['target'])
    feature_names = np.array(diabetes_bunch['feature_names'])
    return feature_array, target_array, feature_names

def calculate_correlation_coeff(feature, target):
    assert(feature.shape == target.shape)
    corr_matrix = np.corrcoef(feature, target)
    return corr_matrix[1,0]

def generate_train_test_idxs(array, split=0.8):
    # Creates 80%, 20% train/test split by default
    split_idx = round(len(array) * 0.8)
    idxs = np.random.permutation(len(array))
    return idxs[:split_idx], idxs[split_idx:]

def fit_linear_model(feature, target):
    # LinearRegression expects 2D arrays, reshape to 2D if 1D
    if (feature.ndim < 2):
        feature = feature.reshape(-1,1)
    if (target.ndim < 2):
        target = target.reshape(-1,1)
    reg = LinearRegression()
    return reg.fit(feature, target)

def main():
    features, target, feature_names = load_diabetes_data()
    coeffs = [calculate_correlation_coeff(feature, target) for feature in features.T]
    max_corr_idx = coeffs.index(max(coeffs, key=abs))

    # Select the feature with the highest linear correlation with the target for the model
    # BMI has the highest correlation: 0.59
    print("{} is the feature with the highest correlation to the target values with a correlation "
          "coefficient of {:.2f}".format(feature_names[max_corr_idx], coeffs[max_corr_idx]))

    model_feature = features[:,max_corr_idx]
    train, test = generate_train_test_idxs(model_feature)
    model = fit_linear_model(model_feature[train], target[train])

    # Make predictions and 
    predictions = model.predict(model_feature[test].reshape(-1,1))
    true_values = target[test]
    mse = mean_squared_error(true_values, predictions)
    print("First 10 predictions of the test set:")
    for i, prediction in enumerate(predictions[:10]):
        print(f"\tPredition: {prediction[0]:.2f},\tActual: {true_values[i]}")
    print(f"Feature coefficient: {model.coef_}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {np.sqrt(mse)}")

    # Create a plot
    plt.scatter(model_feature[test], true_values)
    plt.plot(model_feature[test], predictions, color='black', linewidth=3)
    plt.xlabel(feature_names[max_corr_idx].upper())
    plt.ylabel('Progression')
    plt.title('Diabetes Linear Model')
    plt.show()

if __name__ == '__main__':
    main()
