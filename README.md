# Insurance Cost Prediction Project
By: Muhammad Daffa Al Rasyid


## Overview
This project aims to develop a predictive model for estimating the cost of insurance based on various health-related factors. The dataset used for this project is named Insurance.csv.

## Objective
The primary goal of this program is to provide a tool that allows users to predict insurance costs by leveraging machine learning techniques. The model takes into account features related to the health of the insured individuals.

## Dataset
- Dataset Name: Insurance.csv
- Description: The dataset contains information about individuals, including their age, sex, BMI, number of children, smoking status, region, and insurance charges.
## Libraries Used
The project utilizes the following Python libraries for data analysis, visualization, and machine learning:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import phik
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json


## Workflow
### Data Exploration and Preprocessing:

- The dataset is loaded and explored using Pandas.
- Necessary preprocessing steps are performed, including handling missing values and encoding categorical variables.

### Correlation Analysis:
- Correlation between different features and the insurance charges is visualized using seaborn and matplotlib.

### Model Training:
- The data is split into training and testing sets using train_test_split.
- Standardization is applied to numeric features using StandardScaler.
- Categorical variables are encoded using OrdinalEncoder.
- A Linear Regression model is trained on the preprocessed data.

### Model Evaluation:
- The model is evaluated using Mean Absolute Error (MAE) and R2 score.
### Model Serialization:
- The trained model is saved using the pickle library for future use.
### How to Use
- Ensure that the required libraries are installed.
- Download the Insurance.csv dataset.
- Run the provided Python script, making necessary adjustments if needed.
- Use the trained model to predict insurance costs based on input data.
