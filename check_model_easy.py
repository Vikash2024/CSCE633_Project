from dataloader import *
from models import *
from preprocess import *
from train import rmsre_loss
import sys
def check_loss(val_dataloader):
    '''
    Given a model as command line argument. This function prints the RMSRE of the model.
    '''
    # Model Arch of the best model.
    input_dim = 2  
    model_dim = 128
    num_heads = 2
    num_layers = 3
    additional_features_dim = 1 + 27   #  PCA_VIome
    output_dim = 1  # Lunch calories (regression)
    best_model = TransformerModel(input_dim,model_dim,num_heads,num_layers,additional_features_dim,output_dim)
    best_model.load_state_dict(torch.load('best_model_state/' + sys.argv[1],weights_only=True))
    best_model.eval()

    with torch.no_grad():
        for cgm_sequence, tabular_features, test_labels  in val_dataloader:
            predictions = best_model(cgm_sequence, tabular_features)
           
            print(rmsre_loss(test_labels.numpy(),predictions.squeeze().numpy()))

if __name__ == "__main__":
    cgm_data_test = "cgm_train.csv"
    viome_data_test = "demo_viome_train_processed.csv"
    img_data_test = "img_train.csv"
    labels_test = "label_train.csv"

    val_dataloader= read_dataset(cgm_data_test,
                                        viome_data_test,
                                       img_data_test,
                                        labels_test, flag=False)    # Make sure labels file has Lunch Calories
    
    check_loss(val_dataloader)

   
    
    