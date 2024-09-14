from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score,roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV 

class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str
    ):
        self.clf = estimator
        self.features = features
        self.target = target
        self.optimal_threshold = 0
    

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        # try:
        y_test = df_test[self.target].values
        y_pred_prob = self.clf.predict_proba(df_test[self.features].values)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        optimal_idx = np.argmax(tpr - fpr)
        self.optimal_threshold = thresholds[optimal_idx]
        y_pred_class = (y_pred_prob >= self.optimal_threshold).astype(int)
        self.f1_score = f1_score(y_test,y_pred_class, average='binary')
        self.metric_score = roc_auc_score(y_test,y_pred_class,average = "weighted")
        return {"roc_auc_score": self.metric_score,
                "roc_threshold":self.optimal_threshold,
                "f1_score": self.f1_score
        }
       
        
        # except:    
        #     raise NotImplementedError(
        #         f"You're almost there! Identify an appropriate evaluation metric for your model and implement it here. "
        #         f"The expected output is a dictionary of the following schema: {{metric_name: metric_score}}"
        #     )

    def predict(self, df: pd.DataFrame):
        return (self.clf.predict_proba(df[self.features].values)[:, 1] >= self.optimal_threshold).astype(int)
        # return self.clf.predict_proba(df[self.features].values)[:, 1] 
    
