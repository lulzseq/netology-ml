# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Modelling Helpers
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from  sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score, f1_score

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns