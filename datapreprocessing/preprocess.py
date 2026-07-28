from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



class Data:
    # read the dataset 
    def preprocess_data(self, X):
        
        
        # df = pd.read_csv(path)
        # # seperate dependent and independent columns
        # X = df.drop(['id','diagnosis', 'UnUnnamed: 32'], axis=1)
        # y = df['diagnosis']
    
        # seperate categorical and numerical columns
        cat_cols = X.select_dtypes(include='object').columns
        num_cols = X.select_dtypes(exclude='object').columns
    
        # creating pipeline from the categorical and numerical columns
        cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('Encoder', OneHotEncoder(handle_unknown='ignore')) ])
    
        num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) ]) 
    
    # column transformer for both pipelines
        preprocessor = ColumnTransformer(transformers=[
        ('categorical_pipeline', cat_pipeline, cat_cols),
        ('numerical_pipeline', num_pipeline, num_cols) ])

        # preprocessed_data = preprocessor.fit_transform(X)
        return preprocessor
    # missing values
    
    # scaling
    # encoding
