import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Link for the dataset: https://www.kaggle.com/datasets/m1relly/heart-attack-prediction

# reading the initial dataset extracted from kaggle into a dataframe called "data"
data = pd.read_csv('/content/heart_attack_prediction_dataset.csv')


# describing the dataframe to understand the nature variables
data.head()
data.describe()
