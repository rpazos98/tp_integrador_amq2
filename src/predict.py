"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports

import sys
import logging
import joblib

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


class MakePredictionPipeline:

    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the specified input CSV file and return it as a DataFrame.

        :return: The loaded data as a pandas DataFrame.
        :rtype: pd.DataFrame
        """

        logging.info(f"Reading input data from {self.input_path}")

        data = pd.read_csv(self.input_path)

        return data

    def load_model(self) -> None:
        """
        Load a pre-trained machine learning model from the specified path.
        The loaded model will be stored in the 'model' attribute.
        """
        self.model = joblib.load(self.model_path)

        return None

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
       Make predictions using the loaded machine learning model.

        :param data: The input data for which predictions will be made.
        :type data: pd.DataFrame

        :return: Predicted values based on the input data.
        :rtype: pd.DataFrame
        """

        new_data = self.model.predict(data)

        return new_data

    def write_predictions(self, predicted_data: pd.DataFrame,
                          data: pd.DataFrame) -> None:
        """
       Write the predicted data along with the original input data to an output file.

        :param predicted_data: The DataFrame containing the predicted values.
        :type predicted_data: pd.DataFrame
        :param data: The original input data.
        :type data: pd.DataFrame
        """

        logging.info(f"Writing to {self.output_path}")

        data["prediction"] = predicted_data

        data.to_csv(self.output_path)

        return None

    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds, data)


if __name__ == "__main__":

    logging.info("Starting predict")

    arguments = sys.argv[1:]

    logging.info(arguments)

    input_path = arguments[0]
    output_path = arguments[1]
    model_path = arguments[2]

    logging.info(f"Arguments: {input_path} {output_path} {model_path}")

    pipeline = MakePredictionPipeline(input_path=input_path,
                                      output_path=output_path,
                                      model_path=model_path)
    pipeline.run()
