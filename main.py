import argparse
from preprocess import *
from dataloader import *
from train import *
from hyperparameter import *

device = 'cpu'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSCE633 FINAL PROJECT")
    parser.add_argument('--task', type=str, help="train/test", default = 'train')
    train_dataloader, val_dataloader = read_dataset("cgm_train.csv",
                                                    "demo_viome_train_processed.csv",
                                                    "img_train.csv",
                                                    "label_train.csv")
    best_result = tune_parameters(
                                    train_dataloader=train_dataloader,
                                    val_dataloader=val_dataloader,
                                    lr=0.01,
                                    epochs=500,
                                    input_dim=2,
                                    output_dim=1,
                                    model_dim_list=[16,32,128],
                                    num_heads_list=[2,4],
                                    num_layers_list=[2,3],
                                    additional_features_dim_list=[28]
                                )

    test_dataloader = read_dataset_test("cgm_test.csv",
                                        "demo_viome_test_processed.csv",
                                        "img_test.csv",
                                        "label_test_breakfast_only.csv")
    preds = predict(best_result["best_model"],test_dataloader)
    
    preds_to_csv(preds)