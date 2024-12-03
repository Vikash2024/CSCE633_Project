import torch 
from models import *
from dataloader import *
import copy
import argparse
import os
from tqdm import tqdm 

device = 'cpu'
def rmsre_loss(y_bar, y):
    return ((((y_bar - y) / y_bar) ** 2).mean()) ** 0.5


def rmsre_loss_fn(y_pred, y_true):
    """
    Normalized Root Mean Squared Error (NRMSE) loss for PyTorch tensors.
    
    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): Ground truth values.
    
    Returns:
        torch.Tensor: Computed NRMSE loss.
    """
    # Avoid division by zero by adding a small epsilon to y_true
    epsilon = 1e-8
    nrmse = torch.sqrt(torch.mean(((y_pred - y_true) / (y_true + epsilon)) ** 2))
    return nrmse

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

def trainer(train_dataloader       : torch.utils.data.dataloader,
            val_dataloader         : torch.utils.data.dataloader,
            lr                     : float,
            epochs                 : int, 
            input_dim              : int, 
            model_dim              : int,
            num_heads              : int,
            num_layers             : int,
            additional_features_dim: int,
            output_dim             : int) -> TransformerModel :
    loss_plts_dir = "Train_Test_loss_graphs"
    log_dir = "Model_Logs"
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, additional_features_dim, output_dim)
    criterion = rmsre_loss_fn
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    best_model = None
    
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=epochs)
    min_nrmse = float('inf')
    train_loss_lst = []
    test_loss_lst = []
    os.makedirs(log_dir, exist_ok=True)
    filepath_logs = os.path.join(log_dir, f'Model_{epochs}_{input_dim}_{model_dim}_{num_heads}_{num_layers}_{additional_features_dim}_{output_dim}')
    # Training Loop
    with open(filepath_logs, "w") as log_file:  # Open in write mode
        for epoch in tqdm(range(epochs)): 
            model.train()
            total_train_loss = 0
            total_train_size = 0
            for cgm_sequence, tabular_features, train_labels in train_dataloader:
                # Forward pass
                cgm_sequence, tabular_features, train_labels = cgm_sequence, tabular_features, train_labels
                #print(tabular_features)
                predictions = model(cgm_sequence, tabular_features)
        
                # Compute loss predictions.squeeze(), train_labels

                loss = criterion(predictions.squeeze(), train_labels)
            

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_size += train_labels.shape[0]
                total_train_loss += (loss.item()**2 * train_labels.shape[0])
            
            total_test_loss = 0
            total_test_size = 0
            nrmse = 0
            # Evaluation
            model.eval()
            with torch.no_grad():
                for cgm_sequence, tabular_features , test_labels in val_dataloader:
                    cgm_sequence, tabular_features , test_labels = cgm_sequence, tabular_features, test_labels
                    predictions = model(cgm_sequence, tabular_features )
                    loss = criterion(predictions.squeeze(), test_labels)
                    
                    total_test_size += test_labels.shape[0]
                    total_test_loss += (loss.item() **2 * test_labels.shape[0]) 
                    test_labels = test_labels.cpu()
                    predictions = predictions.cpu()
                    nrmse_val = rmsre_loss(test_labels.numpy(),predictions.squeeze().numpy())

            scheduler.step()
            if (min_nrmse > nrmse_val) :
                best_model = copy.deepcopy(model)
                min_nrmse = nrmse_val
                test_temp = predictions.squeeze().numpy()
                # Prepare the log string
                log_line = (
                    f"Epoch {epoch + 1}, Train Loss: {(total_train_loss / total_train_size) ** 0.5} "
                    f": Test Loss: {(total_test_loss / total_test_size) ** 0.5} : rmsre : {nrmse_val}\n"
                )
                log_file.write(log_line)
                
            
            train_loss_lst.append((total_train_loss / total_train_size) ** 0.5)
            test_loss_lst.append(((total_test_loss / total_test_size) ** 0.5))
        plt.clf()  # Clear the current figure before plotting
        plt.plot(train_loss_lst[50:], label = "train loss")
        plt.plot(test_loss_lst[50:], label = "test loss")
        plt.legend()
        os.makedirs(loss_plts_dir, exist_ok=True)
        filepath_plts = os.path.join(loss_plts_dir, f'Model_{epochs}_{input_dim}_{model_dim}_{num_heads}_{num_layers}_{additional_features_dim}_{output_dim}')
        plt.savefig(filepath_plts)
        return best_model
    
def predict(best_model, test_dataloader):
    # Evaluation
    best_model.eval()
    preds = []
    with torch.no_grad():
        for cgm_sequence, tabular_features, test_labels  in test_dataloader:
            predictions = best_model(cgm_sequence, tabular_features)
            preds.extend(predictions.squeeze().tolist())
    return preds
def preds_to_csv(preds):
    output = pd.DataFrame(preds)
    output.columns = ['preds']
    output['row_id'] = range(0, len(output))
    output = output[['row_id','preds']]
    output.to_csv("submissions.csv", index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSCE633 FINAL PROJECT")
    parser.add_argument('--runtime',  default = 'testrun')
    args = parser.parse_args()
    runtime = args.runtime
    train_dataloader, val_dataloader = read_dataset("cgm_train.csv",
                                                    "demo_viome_train_processed.csv",
                                                    "img_train.csv",
                                                    "label_train.csv")
   
    # Model Parameters
    input_dim = 2  # CGM data has 2 features: timeInTicks and glucose
    model_dim = 64
    num_heads = 4
    num_layers = 3
    additional_features_dim = 1 + 27   #  PCA_VIome
    output_dim = 1  # Lunch calories (regression)
    epoch = 500
    lr = 0.01
    best_model = trainer(train_dataloader, val_dataloader,lr,epoch,input_dim,model_dim,num_heads,num_layers,additional_features_dim,output_dim)

