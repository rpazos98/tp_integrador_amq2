import subprocess

subprocess.run(['python', 'src/feature_engineering.py', 'data/Train_BigMart.csv' , 'data/train_final.csv', 'data/Test_BigMart.csv', 'data/test_final.csv'])
subprocess.run(['python', 'src/train.py', 'data/train_final.csv', 'data/model.pkl'])