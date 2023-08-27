import os
import sys
import pickle

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")

            # Define which columns should be ordinal encoded and which should be scaled
            # In this dataset their is no categorical columns is present so no required

            num_columns = ['Followers']

            cat_columns = ['USERNAME', 'Hashtags']

            logging.info("Pipeline Initaited")

            # Numerical Pipeline

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())

                ]

            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(handle_unknown =  'use_encoded_value', unknown_value = -1)),
                ]
            )

            # Preprocessor

            preprocessor = ColumnTransformer(
            [
            ('cat_pipeline', cat_pipeline, cat_columns),
            ('num_pipeline', num_pipeline, num_columns)
        
                ],
            remainder = 'passthrough'
            )

            final_pipeline = Pipeline(
                steps = [
            ('preprocessor', preprocessor),
            ]
            )

            return final_pipeline

            logging.info("Pipeline Completed")

        except Exception as e:
            logging.info("Error In Data Transformation")
            raise CustomException(e, sys)
        
    # Read Train & Test Data

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading Train & Test Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train & Test Data Completed")
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")
            logging.info("Obtaining Processing Object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time since posted (hours)'

            drop_columns = [target_column_name, 'Caption', 'Likes', 'Unnamed: 0']

            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1)

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_columns, axis = 1)

            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Input_feature_train_df Head:\n{input_feature_train_df.head().to_string()}")
            logging.info(f"Target_feature_train_df Head:\n{target_feature_train_df.head().to_string()}")
            logging.info(f"Input_feature_test_df Head:\n{input_feature_test_df.head().to_string()}")
            logging.info(f"Target_feature_test_df Head:\n{target_feature_test_df.head().to_string()}")

            # Transforming using Preprocessor Obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying Preprocessing Object on Training & Testing")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info(f"Train_arr Dataframe Head:\n{train_arr}")
            logging.info(f"Test_arr Dataframe Head:\n{test_arr}")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor pickle file saved")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Exception Occured in the Initiate_Datatransformation")
            raise CustomException(e,sys)
        




if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        

        



        

        