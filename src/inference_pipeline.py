import subprocess

subprocess.run(['python', 'feature_engineering.py', '-p../data/Train_BigMart.csv ../data/train_final.csv'])
subprocess.run(['python', 'feature_engineering.py', '-p../data/Test_BigMart.csv ../data/test_final.csv'])

# subprocess.run(['python', 'predict.py'])