
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datapreprocessing.preprocess import Data

PATH="data/Cancer_Data.csv"
data = pd.read_csv(PATH)

X = data.drop(['id','diagnosis', 'Unnamed: 32'], axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = Data()

X_train = preprocess.preprocess_data(X_train)

print(X_train)
