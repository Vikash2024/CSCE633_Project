# CSCE 633 Final Project 
This is the repository for the final project submission for CSCE 633. The authors of this repository are Arun Akash Rangaraj, Julian Ezra Samuel, and Vikash Sunkara. 

This repository is used to preprocess and use a nutrition dataset to predict the calories consumed for lunch. 

## How to use this repository?

Ensure to have the following files in the same directory: cgm_train.csv, cgm_test.csv, img_train.csv, img_test.csv, label_train.csv, demo_viome_train.csv and demo_viome_test.csv. 

First, process the viome data. 

'''bash
python viome_process.py
'''

Running the command above will give out two csv files: demo_viome_train_processed.csv and demo_viome_test_processed.csv. 

To train and find the best model and test it on the test set, use the following command. The best model weights will be stored n the folder 'best_model_state'

'''bash
python main.py
'''

In order test an already pre trained model, use the following command and ensure to enter the model name. For example, if the 

'''bash
python check_model_easy.py [model_name]
'''




