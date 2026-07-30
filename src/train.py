
from sklearn.model_selection import train_test_split 
import pandas as pd 
import os 
import sys 
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datapreprocessing.preprocess import Data 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

#pip install xgboost
PATH="data/Cancer_Data.csv"
data =pd.read_csv(PATH)

X = data.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = Data()
preprocessor_transformer = preprocessor.preprocess_data(X_train)

X_train = preprocessor_transformer.fit_transform(X_train)


models = {
    "Logistic Regression": LogisticRegression(max_iter=100),
    "Decision Tree" : DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(), 
    
}
X_test = preprocessor_transformer.transform(X_test)
for name, algo in models.items():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Cancer_Detection_Experiment")
    
    # From here the model experiment is starting..
    with mlflow.start_run(run_name=name):
        
        algo.fit(X_train, y_train)
        
        y_pred = algo.predict(X_test)
        
        # print("Unique predictions:", set(y_pred))
        # print("Prediction counts:")
        # print(pd.Series(y_pred).value_counts())

        # print("\nActual counts:")
        # print(y_test.value_counts())



        # print("\nFirst 20 Predictions:")
        # print(y_pred[:20])

        # print("\nFirst 20 Actual:")
        # print(y_test.iloc[:20].values)
        
        
        # Here we are calculatiing the metrics of the model 
        acc_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='M')
        recall = recall_score(y_test, y_pred, pos_label='M')
        score = f1_score(y_test, y_pred, pos_label='M')

        # Storing the model metrics 
        mlflow.log_metric("accuracy", acc_score)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("F1 Score", score)

        # Logging the model 
        mlflow.sklearn.log_model(algo, name=name)

        # message 
        print(f'{name} has been saved successfully!')