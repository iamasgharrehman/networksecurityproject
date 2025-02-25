from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation,DataValidationArtifact
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.entity.config_entity import DataTransformationConfig,ModelTrainerConfig
from networksecurity.components.data_transformation import DataTransformation,DataTransformationArtifact
import sys

if __name__ =="__main__":
    try:

        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion.")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)

        data_validation_config=DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("initiate data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data validation completed")
        print(data_validation_artifact)

        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Initiate the Data transformation")
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data transformation completed")

        logging.info("model training started")
        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config,data_transformation_artifact)
        model_trainer.initiate_model_trainer()
        logging.info('Model training artifacts created')
    except Exception as e:
        raise NetworkSecurityException(e,sys)

