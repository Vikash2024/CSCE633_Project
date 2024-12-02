# CGM embedding model ()
# image embeddeding
# tabular embedding 

# Caloriepredictor(str, str, str)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.models as models

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, additional_features_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim

        # Input embedding for CGM data
        self.embedding = nn.Linear(input_dim, model_dim)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        self.transformer_out_fc = nn.Linear(model_dim, model_dim )
        self.transformer_out_bn = nn.BatchNorm1d(model_dim )
        # Fully connected layers for tabular features
        self.additional_fc = nn.Linear(additional_features_dim, model_dim )
        self.additional_fc_bn =  nn.BatchNorm1d(model_dim)

       

        # Final output layer
        self.fc1 = nn.Linear(2 * model_dim, 2 * model_dim // 2) 
        self.fc2 = nn.Linear(2 * model_dim // 2, 2 * model_dim // 4) 
        self.fc3 = nn.Linear(2 * model_dim // 4, 4) 
        self.fc4 = nn.Linear(4, output_dim) 
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(2 * model_dim // 2)
        self.bn2 = nn.BatchNorm1d(2 * model_dim // 4)
        self.bn3 = nn.BatchNorm1d(4)
        
    def forward(self, cgm_sequence, tabular_features):
        # Embedding for CGM data
        cgm_embedded = self.embedding(cgm_sequence)  # (batch_size, seq_len, model_dim)
        
        # Reshape for transformer: (seq_len, batch_size, model_dim)
        cgm_embedded = cgm_embedded.permute(1, 0, 2)  # (seq_len, batch_size, model_dim)
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(cgm_embedded)  # (seq_len, batch_size, model_dim)

        
        # Take the output of the last timestep
        transformer_output_last = transformer_output[transformer_output.shape[0] // 2, :, :]  # (batch_size, model_dim)
        
        # Process tabular features
        
        additional_features_out = self.additional_fc(tabular_features)
        
        additional_features_out = self.additional_fc_bn(additional_features_out.squeeze(1))  
        
        additional_features_out = self.relu(additional_features_out)                       
        
        additional_features_out = self.dropout(additional_features_out)                     # (batch_size, model_dim)
       

        # Concatenate transformer output, tabular features, and image features
        combined_features = torch.cat((transformer_output_last, additional_features_out), dim=-1)  # (batch_size, 2 * model_dim)

        # Final prediction
        x = self.fc1(combined_features)  # (batch_size, output_dim)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x
