import subprocess

subprocess.run(['python',
                'src/predict.py',
                'src/script_generated/test_final.csv',
                'src/script_generated/data_test_predicted',
                'src/script_generated/model.pkl'])
