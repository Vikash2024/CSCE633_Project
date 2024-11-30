import torch 
from models import *
from dataloader import *
import copy
import argparse
import os
from tqdm import tqdm 
input_dim = 2  # CGM data has 2 features: timeInTicks and glucose
model_dim = 64
num_heads = 4
num_layers = 10
additional_features_dim = 1 + 27   #  PCA_VIome
output_dim = 1  # Lunch calories (regression)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def nrmse_loss(y_bar, y):
    return ((((y_bar - y) / y_bar) ** 2).mean()) ** 0.5
def nrmse_loss_fn(y_pred, y_true):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSCE633 FINAL PROJECT")
    parser.add_argument('--runtime',  default = 'testrun')
    args = parser.parse_args()
    runtime = args.runtime
    train_dataloader, val_dataloader = read_dataset()
    model = TransformerModel(input_dim, model_dim, num_heads, num_layers, additional_features_dim, output_dim).to(device)
    criterion = nrmse_loss_fn
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_model = None
    epochs = 500
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=epochs)
    min_nrmse = float('inf')
    train_loss_lst = []
    test_loss_lst = []
    # Training Loop
    for epoch in tqdm(range(epochs)): 
        model.train()
        total_train_loss = 0
        total_train_size = 0
        for cgm_sequence, tabular_features, train_labels in train_dataloader:
            # Forward pass
            cgm_sequence, tabular_features, train_labels = cgm_sequence.to(device), tabular_features.to(device), train_labels.to(device)
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
            total_train_loss += (loss.item()**2) * train_labels.shape[0]

        
        total_test_loss = 0
        total_test_size = 0
        nrmse = 0
        # Evaluation
        model.eval()
        with torch.no_grad():
            for cgm_sequence, tabular_features , test_labels in val_dataloader:
                cgm_sequence, tabular_features , test_labels = cgm_sequence.to(device), tabular_features.to(device) , test_labels.to(device)
                predictions = model(cgm_sequence, tabular_features )
                loss = criterion(predictions.squeeze(), test_labels)
                
                total_test_size += test_labels.shape[0]
                total_test_loss += (loss.item() ) * test_labels.shape[0]
                test_labels = test_labels.cpu()
                predictions = predictions.cpu()
                nrmse_val = nrmse_loss(test_labels.numpy(),predictions.squeeze().numpy())

        scheduler.step()
        if (min_nrmse > nrmse_val) :
            best_model = copy.deepcopy(model)
            min_nrmse = nrmse_val
            test_temp = predictions.squeeze().numpy()
            print(f"Epoch {epoch+1}, Train Loss: {(total_train_loss / total_train_size) ** 0.5} : Test loss : {(total_test_loss / total_test_size) ** 0.5} : nrmse : {nrmse_val}" )
        
        train_loss_lst.append((total_train_loss / total_train_size) ** 0.5)
        test_loss_lst.append(((total_test_loss / total_test_size) ** 0.5))
    plt.plot(train_loss_lst[50:], label = "train loss")
    plt.plot(test_loss_lst[50:], label = "test loss")
    plt.legend()
    plt.savefig(f'{runtime}.png')
    