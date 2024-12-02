from dataloader import *
from models import *
from preprocess import *
from train import rmsre_loss
def check_loss(val_dataloader):
    input_dim = 2  # CGM data has 2 features: timeInTicks and glucose
    model_dim = 64
    num_heads = 4
    num_layers = 2
    additional_features_dim = 1 + 27   #  PCA_VIome
    output_dim = 1  # Lunch calories (regression)
    best_model = TransformerModel(input_dim,model_dim,num_heads,num_layers,additional_features_dim,output_dim)
    best_model.load_state_dict(torch.load('best_model_state/Best_Model_500_64_2_4_2_28',weights_only=True))
    best_model.eval()

    with torch.no_grad():
        for cgm_sequence, tabular_features, test_labels  in val_dataloader:
            predictions = best_model(cgm_sequence, tabular_features)
           
            print(rmsre_loss(test_labels.numpy(),predictions.squeeze().numpy()))

if __name__ == "__main__":
    cgm_data = "cgm_train.csv"
    viome_data = "demo_viome_train_processed.csv"
    img_data = "img_train.csv",
    labels = "label_train.csv"

    val_dataloader= read_dataset("cgm_train.csv",
                                        "demo_viome_train_processed.csv",
                                        "img_train.csv",
                                        "label_train.csv",flag = False)
    check_loss(val_dataloader)
    