import subprocess

subprocess.run(['python', 'src/feature_engineering_training.py', 'data/Train_BigMart.csv' , 'src/script_generated/train_final.csv', 'data/Test_BigMart.csv', 'src/script_generated/test_final.csv'])
subprocess.run(['python', 'src/train.py', 'src/script_generated/train_final.csv', 'src/script_generated/model.pkl'])