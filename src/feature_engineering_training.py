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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineeringPipeline(object):

    def __init__(self, input_path_train, output_path_train, input_path_test, output_path_test):
        self.input_path_train = input_path_train
        self.output_path_train = output_path_train
        self.input_path_test = input_path_test
        self.output_path_test = output_path_test

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
            
        # COMPLETAR CON CÓDIGO\

        logging.info(f"Reading input data from {self.input_path_test} {self.input_path_train}")

        data_train = pd.read_csv(self.input_path_train)
        data_test = pd.read_csv(self.input_path_test)
        # Identificando la data de train y de test, para posteriormente unión y separación
        data_train['Set'] = 'train'
        data_test['Set'] = 'test'

        data = pd.concat([data_train, data_test], ignore_index=True, sort=False)

        return data

    
    def data_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """

        logging.info(f"Applying transformation")

        # data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']

        # data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
       
        # productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
        # for producto in productos:
        #     moda = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
        #     data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda  

        # outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        
        # for outlet in outlets:
        #     data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'
        
        # data.loc[data['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
        # data.loc[data['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
        # data.loc[data['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
        # data.loc[data['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
        # data.loc[data['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'

        # data['Item_Type'] = data['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
        # 'Seafood': 'Meats', 'Meat': 'Meats',
        # 'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
        # 'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
        # 'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

        # # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        # data.loc[data['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        # data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])

        # dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()

        # dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        # dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos

        # dataframe = pd.get_dummies(dataframe, columns=['Outlet_Type'])

        # # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
        # dataset = dataframe.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        dataset = transform(data)

        return dataset 

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO

        # División del dataset de train y test
        df_train = transformed_dataframe.loc[transformed_dataframe['Set'] == 'train']
        df_test = transformed_dataframe.loc[transformed_dataframe['Set'] == 'test']

        # Eliminando columnas sin datos
        df_train.drop(['Set'], axis=1, inplace=True)
        df_test.drop(['Item_Outlet_Sales','Set'], axis=1, inplace=True)

        logging.info(f"Writing output data to {self.output_path_train} {self.output_path_test}")

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

    logging.info(f"Arguments: {input_path_train} {output_path_train} {input_path_test} {output_path_test}")

    FeatureEngineeringPipeline(input_path_train = input_path_train , output_path_train = output_path_train, 
                               input_path_test = input_path_test, output_path_test = output_path_test).run()  