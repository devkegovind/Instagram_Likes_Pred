import os
import sys
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
        

        



        

        