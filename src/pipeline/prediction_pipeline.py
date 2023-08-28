import sys
import os
import pandas as pd
import numpy as np

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info("Exception Occured in Prediction")
            raise CustomException (e, sys)
        
class CustomData:
    def __init__(self, 
                 USERNAME:str,
                 Followers:int,
                 Hashtags: str):
        self.USERNAME = USERNAME
        self.Followers = Followers
        self.Hashtags = Hashtags

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'USERNAME' : [self.USERNAME],
                'Followers' : [self.Followers],
                'Hashtags' : [self.Hashtags]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info("Exception Occured in Prediction Pipeline")
            raise CustomException(e,sys)