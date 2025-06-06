# House Prices Prediction - Linear Regression (Student Style, Best Practice)

This project uses **Linear Regression** to predict house prices in Ames, Iowa based on housing characteristics. It follows a clean and reproducible workflow with preprocessing, scaling, and feature encoding.

---

## Project Overview

This model predicts the final sale price (`SalePrice`) of homes using features such as lot size, quality ratings, square footage, and more. The aim is to build a simple, interpretable baseline model using **linear regression** with proper preprocessing and evaluation.

---

## Repository Structure
├── train.csv # Training data
├── test.csv # Test data
├── Complete Analysis.py # Main script
├── submission.csv # Submission file (Kaggle format)
├── README.md # Documentation
└── requirements.txt # Python dependencies

---

## Problem Statement

Accurately predicting house prices is valuable for:
- Home buyers and investors
- Real estate professionals
- Mortgage risk assessment

The dataset contains 79 features describing every aspect of residential homes in Ames, Iowa.

---

## Requirements
Python 3.10+
pandas
numpy
matplotlib
seaborn
scikit-learn
See requirements.txt for full list.

## Key Learnings
Preprocessing (missing values + one-hot encoding) is critical for linear models.
Scaling numeric values improves convergence and comparability of coefficients.
Even a simple linear regression can provide surprisingly strong performance when combined with good feature engineering.

## Limitations & Improvements
Currently uses only linear regression — nonlinear models like XGBoost or Lasso could improve performance.

## Author
A1C Sik Kiu AU
Data Analytics Graduate Student 


