"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports

import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
            
        # COMPLETAR CON CÓDIGO\

        logging.info(f"Reading input data from {self.input_path}")

        pandas_df = pd.read_csv(self.input_path)
        return pandas_df

    
    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """

        logging.info(f"Applying transformation")


        data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']
        data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
        
        productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for producto in productos:
            moda_opt = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode()
            if len(moda_opt) != 0:
                moda = moda_opt.iloc[0,0]
                data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda

        outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())

        for outlet in outlets:
            data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        data.loc[data['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'

        data['Item_Type'] = data['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
            'Seafood': 'Meats', 'Meat': 'Meats',
            'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
            'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
            'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

        data.loc[data['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])

        data = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()

        data['Outlet_Size'] = data['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        data['Outlet_Location_Type'] = data['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})

        data = pd.get_dummies(data, columns=['Outlet_Type'])

        data = data.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        return data

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO

        logging.info(f"Writing output data to {self.output_path}")

        transformed_dataframe.to_csv(self.output_path)
        
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":

    logging.info("Starting feature engineering")

    parser = argparse.ArgumentParser(description='Feature engineering script that applies the necessary transformations to the dataset')

    parser.add_argument('-p', '--input', help='Input path for the dataset')

    args = parser.parse_args().input.split(" ")

    logging.info(args)

    FeatureEngineeringPipeline(input_path = args[0],
                               output_path = args[1])\
                                .run()  