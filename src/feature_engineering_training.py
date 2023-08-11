"""
feature_engineering.py

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureEngineeringPipeline(object):

    def __init__(self, input_path_train, output_path_train,
                 input_path_test, output_path_test):
        self.input_path_train = input_path_train
        self.output_path_train = output_path_train
        self.input_path_test = input_path_test
        self.output_path_test = output_path_test

    def read_data(self) -> pd.DataFrame:
        """

        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """

        # COMPLETAR CON CÓDIGO\

        logging.info(
            f"Reading input data from {self.input_path_test} {self.input_path_train}")

        data_train = pd.read_csv(self.input_path_train)
        data_test = pd.read_csv(self.input_path_test)
        # Identificando la data de train y de test, para posteriormente unión y
        # separación
        data_train['Set'] = 'train'
        data_test['Set'] = 'test'

        data = pd.concat([data_train, data_test],
                         ignore_index=True, sort=False)

        return data

    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data (pd.DataFrame): A DataFrame containing the input data.

        dataset pd.DataFrame: A new DataFrame containing the transformed data.

        """

        logging.info(f"Applying transformation")

        dataset = transform(data)

        return dataset

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Write the prepared data (transformed DataFrame) to output files for train and test sets.

        :param transformed_dataframe: The DataFrame containing the prepared data.
        :type transformed_dataframe: pd.DataFrame
        """

        # COMPLETAR CON CÓDIGO

        # División del dataset de train y test
        df_train = transformed_dataframe.loc[transformed_dataframe['Set'] == 'train']
        df_test = transformed_dataframe.loc[transformed_dataframe['Set'] == 'test']

        # Eliminando columnas sin datos
        df_train.drop(['Set'], axis=1, inplace=True)
        df_test.drop(['Item_Outlet_Sales', 'Set'], axis=1, inplace=True)

        logging.info(
            f"Writing output data to {self.output_path_train} {self.output_path_test}")

        df_train.to_csv(self.output_path_train)
        df_test.to_csv(self.output_path_test)

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

    input_path_test = arguments[2]
    output_path_test = arguments[3]

    logging.info(
        f"Arguments: {input_path_train} {output_path_train} {input_path_test} {output_path_test}")

    FeatureEngineeringPipeline(input_path_train=input_path_train, output_path_train=output_path_train,
                               input_path_test=input_path_test, output_path_test=output_path_test).run()
