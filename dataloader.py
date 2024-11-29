from preprocess import *

class CGMData(Dataset):
    def __init__(self, x_train: pd.DataFrame, labels: float):
        """
        Args:
            x_train (pd.DataFrame): DataFrame with columns:
                - 'Breakfast Calories': float
                - 'PCA_1 .. PCA_n': float
                - 'CGM Data': list of (timeInTicks, glucose)
                - 'Image Before Lunch' : Image array.
            labels - Lunch Calories : float.
        """
        self.x_train = x_train
        self.labels = labels
        self.pca_columns = [col for col in x_train.columns if col.startswith('PCA_')]

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        # Extract row
        row = self.x_train.iloc[idx]

        # Process CGM Data
        sequence = row['CGM Data']                                          # list of (timeInTicks, glucose)
        times, glucose = zip(*sequence)
        times = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)      # Shape: (seq_len, 1)
        glucose = torch.tensor(glucose, dtype=torch.float32).unsqueeze(-1)  # Shape: (seq_len, 1)
        combined_sequence = torch.cat((times, glucose), dim=-1)             # Shape: (seq_len, 2)
        
        
        tabular_features = torch.tensor([
            [row[col] for col in self.pca_columns] + [row['Breakfast Calories']]
        ], dtype=torch.float32)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        lunch_image = row['Image Before Lunch']
        lunch_image = torch.tensor(lunch_image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)

        return combined_sequence, tabular_features, lunch_image , label
    
def read_dataset():
    # Load Data from CSV file.
    train_data = pd.read_csv('cgm_train.csv')
    train_labels = pd.read_csv('label_train.csv')
    viome_data_train = pd.read_csv("demo_viome_train_processed.csv").drop("Unnamed: 0", axis=1)
    image_data_train = pd.read_csv("img_train.csv").drop('Image Before Breakfast', axis = 1)

    # Merge Data to a single dataframe
    train_data = pd.merge(train_data, viome_data_train, how="left")
    train_data = pd.merge(train_data, image_data_train, how= "left")
    train_data = pd.concat([train_data, train_labels['Breakfast Calories']], axis=1)
    train_data = pd.concat([train_data, train_labels['Lunch Calories']], axis=1)

    # Preprocessing
    train_data = removeNullRows(train_data)                                 
    train_data['CGM Data'] =  train_data['CGM Data'].apply(lambda x: eval(x))
    train_data['Image Before Lunch'] = train_data['Image Before Lunch'].apply(lambda x : np.array(eval(x)))
    train_data = convertTimeStrToTicks(train_data)                                                  
    train_data['CGM Data'] = train_data['CGM Data'].apply(lambda x : paddingInsideWithMean(x,300))
    train_data.drop(['Breakfast Time','Lunch Time'],axis = 1,inplace= True)
    cgm_sequences = train_data['CGM Data']
    cgm_sequences = pad_cgm_sequences(cgm_sequences)
    cgm_sequences  = pd.DataFrame({'CGM Data': [cgm_sequences[i] for i in range(cgm_sequences.shape[0])]})
    train_data.drop('CGM Data', axis=1,inplace=True)
    train_data = pd.concat([train_data,cgm_sequences], axis=1)

    # Train / Validation Split.
    x_train = train_data.iloc[:,2:].drop('Lunch Calories',axis=1)
    y_train = train_data['Lunch Calories']

    x_train, x_val, y_train, y_val = train_test_split(
    x_train,         # Features DataFrame
    y_train,         # Label
    test_size=0.1,   # Fraction of data for validation
    random_state=42  # Random seed for reproducibility
    )
    x_train = x_train.reset_index(drop=True)
    x_val   = x_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val =   y_val.reset_index(drop=True)
    train_dataset = CGMData(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = CGMData(x_val, y_val)
    val_dataloader =  DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    train_dataloader, val_dataloader = read_dataset()
    for cgm_sequence, tabular_features, img_feature ,train_labels in train_dataloader:
        print(f'train_labels : {cgm_sequence}' )
        