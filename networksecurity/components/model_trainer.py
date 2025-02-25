import os
import sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

import mlflow
from urllib.parse import urlparse


from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.utlis import save_object, load_object,load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

import dagshub
dagshub.init(repo_owner='iamasgharrehman', repo_name='networksecurityproject', mlflow=True)

class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact) -> ModelTrainerArtifact:
    
        try:
            self.model_trianer_config=model_trainer_config
            self.data_trnasformation_artifact =data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,classification_metric):
        with mlflow.start_run():
            f1_score=classification_metric.f1_score
            precision_score=classification_metric.precision_score
            recall_score=classification_metric.recall_score
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric('recall_score', recall_score)
            mlflow.sklearn.log_model(best_model,'model')
        

    def train_model(self,train_X,train_y,test_X,test_y):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }

        model_report:dict=evaluate_models(train_X=train_X,train_y=train_y,
                        test_X=test_X,test_y=test_y,
                        models=models,param=params)


        ## To get the best model score form dict
        best_model_score = max(sorted(model_report.values()))


        # to get the best model name
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]
        y_train_pred=best_model.predict(train_X)
        classification_train_metric=get_classification_score(train_y,y_train_pred)


        # Track experiments with mlflow
        self.track_mlflow(best_model,classification_train_metric)


        y_test_pred = best_model.predict(test_X)
        classification_test_metric = get_classification_score(test_y,y_test_pred)

        # Track Experiment with Mlfow
        self.track_mlflow(best_model,classification_test_metric)


        #model pusher
        save_object('final_model/model.pkl', best_model)

        preprocessor = load_object(file_path=self.data_trnasformation_artifact.transformed_object_file_path)

        model_dir_path=os.path.dirname(self.model_trianer_config.model_trainer_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        # save_object(model_dir_path,best_model)

        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        # save_object(file_path=model_dir_path,obj=Network_Model)



        #Model Trainer Aritfact
        model_trainer_artifact=ModelTrainerArtifact(
            trainer_model_file_path=self.model_trianer_config.model_trainer_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact:{model_trainer_artifact}")
        return model_trainer_artifact

        

    def initiate_model_trainer(self):
        try:
            train_array=load_numpy_array_data(self.data_trnasformation_artifact.transformed_test_file_path)
            test_array=load_numpy_array_data(self.data_trnasformation_artifact.transformed_test_file_path)

            train_X,train_y,test_X,test_y=(
                train_array[:,: -1],
                train_array[:, -1],
                test_array[:,:-1],
                test_array[:, -1]
            )
            model_trainer_artifact=self.train_model(train_X,train_y,test_X,test_y)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)

