"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from scipy import stats
import logging
import joblib

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """

        logging.info(f"Reading input data from {self.input_path}")

        data = pd.read_csv(self.input_path)

        return data

    def load_model(self) -> None:
        """
        COMPLETAR DOCSTRING
        """    
        self.model = joblib.load(self.model_path)
        
        return None


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """
   
        new_data = self.model.predict(data)

        return new_data


    def write_predictions(self, predicted_data: pd.DataFrame, data: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        """

        data["prediction"] = predicted_data

        data.to_csv(self.output_path)

        return None

    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds, data)


if __name__ == "__main__":

    logging.info("Starting feature engineering")

    arguments = sys.argv[1:]

    logging.info(arguments)

    input_path = arguments[0]
    output_path = arguments[1]
    model_path = arguments[2]

    logging.info(f"Arguments: {input_path} {output_path} {model_path}")
    
    pipeline = MakePredictionPipeline(input_path = input_path,
                                      output_path = output_path,
                                      model_path = model_path)
    pipeline.run()  