"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports

import logging
import pandas as pd
from joblib import dump
import sys

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self, input_path) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING

        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """

        logging.info(f"Reading from {input_path}")

        pandas_df = pd.read_csv(input_path)
        return pandas_df

    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train a machine learning model using the provided DataFrame and return the trained model.

        :param df: The DataFrame containing the training data.
        :type df: pd.DataFrame

        :return: The trained machine learning model.
        :rtype: object

        """

        # COMPLETAR CON CÓDIGO

        logging.info("Training model")

        seed = 28
        model = LinearRegression()

        # División de dataset de entrenaimento y validación
        X = df.drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(
            X, df['Item_Outlet_Sales'], test_size=0.3, random_state=seed)

        # Entrenamiento del modelo
        model.fit(x_train, y_train)

        # Predicción del modelo ajustado para el conjunto de validación
        pred = model.predict(x_val)

        # Cálculo de los errores cuadráticos medios y Coeficiente de
        # Determinación (R^2)
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        R2_train = model.score(x_train, y_train)
        print('Métricas del Modelo:')
        print(
            'ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

        mse_val = metrics.mean_squared_error(y_val, pred)
        R2_val = model.score(x_val, y_val)
        print(
            'VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

        print('\nCoeficientes del Modelo:')
        # Constante del modelo
        print('Intersección: {:.2f}'.format(model.intercept_))

        # Coeficientes del modelo
        coef = pd.DataFrame(x_train.columns, columns=['features'])
        coef['Coeficiente Estimados'] = model.coef_
        (coef, '\n')

        return model

    def model_dump(self, model_trained, output_path) -> None:
        """
        Save a trained machine learning model to the specified output path.

        :param model_trained: The trained machine learning model.
        :param output_path: The path where the model will be saved.
        :type output_path: str
        """

        # COMPLETAR CON CÓDIGO

        logging.info(f"Dumping model to {output_path}")

        dump(model_trained, output_path)

    def run(self):

        df = self.read_data(self.input_path)
        model_trained = self.model_training(df)
        self.model_dump(model_trained, self.model_path)


if __name__ == "__main__":

    logging.info("Starting training")

    arguments = sys.argv[1:]

    input_path = arguments[0]
    output_path = arguments[1]

    logging.info(f"Input path {input_path} Output path {output_path}")

    ModelTrainingPipeline(input_path=input_path,
                          model_path=output_path).run()
