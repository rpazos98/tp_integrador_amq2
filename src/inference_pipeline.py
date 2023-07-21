import subprocess

# subprocess.run(['python', 'src/feature_engineering_inference.py', 'data/Test_BigMart.csv' , 'Notebook/data_test_transformed'])
subprocess.run(['python', 'src/predict.py', 'data/test_final.csv', 'Notebook/data_test_predicted', 'data/model.pkl'])