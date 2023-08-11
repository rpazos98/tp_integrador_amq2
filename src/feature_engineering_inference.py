"""
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
from feature_engineering_logic import transform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """    
        :return pandas_df: The desired DataLake table as a DataFrame. 
        :rtype: pd.DataFrame.
        """
            
        # COMPLETAR CON CÓDIGO\

        logging.info(f"Reading input data from {self.input_path} {self.output_path}")

        data = pd.read_csv(self.input_path)

        return data

    
    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        :param data: The input DataFrame to be transformed.
        :type data: pd.DataFrame.
        
        :return: The transformed DataFrame.
        :rtype: pd.DataFrame.
        """

        logging.info(f"Applying transformation")

        dataset = transform(data)

        return dataset 

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Write the transformed DataFrame to an output file.
        
        :param transformed_dataframe: The DataFrame containing the prepared data.
        :type transformed_dataframe: pd.DataFrame
        """
        
        # COMPLETAR CON CÓDIGO

        logging.info(f"Writing output data to {self.output_path}")

        transformed_dataframe.to_csv(self.output_path_train)
                
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":

    logging.info("Starting feature engineering")

    arguments = sys.argv[1:]

    logging.info(arguments)

    input_path_train = arguments[0]
    output_path_train = arguments[1]

    logging.info(f"Arguments: {input_path_train} {output_path_train}")

    FeatureEngineeringPipeline(input_path = input_path_train , output_path = output_path_train).run()  