import sys
import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constants import training_pipeline
from networksecurity.entity.artifact_entity import (
    DataValidationArtifact, DataTransformationArtifact
)
from networksecurity.utils.main_utils.utlis import save_numpy_array_data, save_object
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationArtifact):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def get_data_transformation_object(cls)->Pipeline:
        """
        It initiaises a KNNImputer object with the with the parameters defined in training_pipelin
        and return a Pipeline object with the KNNImputer object as the first step.

        Args:
        cls: DataTransformation

        Returns:
        A Pipeline Object
        """

        logging.info(
            "Entered get_data_transformer object method of Transformation class"
        )
        try:
            imputer:KNNImputer=KNNImputer(**training_pipeline.DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Intialise KNNImputer with {training_pipeline.DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor:Pipeline=Pipeline([
                ('imputer', imputer)
            ])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("entered intiate_data_transformation methof of transformation class")
        try:
            logging.info("strating data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## training dataframe
            input_feature_train_df=train_df.drop(columns=[training_pipeline.TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[training_pipeline.TARGET_COLUMN]
            target_feature_train_df=target_feature_train_df.replace(-1,0)

            # testing dataframe
            input_feature_test_df=test_df.drop(columns=[training_pipeline.TARGET_COLUMN],axis=1)
            target_feature_test_df=test_df[training_pipeline.TARGET_COLUMN]
            target_feature_test_df=target_feature_test_df.replace(-1,0)

            preprocessor=self.get_data_transformation_object()

            preprocessor_obj=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr=np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # save numpy array data
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor_obj)
            
            save_object('final_model/preprocessor.pkl', preprocessor_obj)

                    #preparing artifacts
            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
