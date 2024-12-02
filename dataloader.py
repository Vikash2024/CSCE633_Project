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

        return combined_sequence, tabular_features, label
    
def read_dataset(cgm_data,viome_data,img_data,label):
    # Load Data from CSV file.
    train_data = pd.read_csv(cgm_data)
    train_labels = pd.read_csv(label)
    viome_data_train = pd.read_csv(viome_data).drop("Unnamed: 0", axis=1)
    image_data_train = pd.read_csv(img_data).drop('Image Before Breakfast', axis = 1)

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

def read_dataset_test(cgm_data,viome_data,img_data,label):
    test_data = pd.read_csv(cgm_data)
    test_labels = pd.read_csv(label)
    viome_data_test = pd.read_csv(viome_data).drop("Unnamed: 0", axis=1)
    image_data_test = pd.read_csv(img_data).drop('Image Before Breakfast', axis = 1)

    test_data = pd.merge(test_data, viome_data_test, how="left", on="Subject ID")
    test_data = pd.merge(test_data, image_data_test, how= "left")
    test_data = pd.concat([test_data, test_labels['Breakfast Calories']], axis=1)

    test_data = removeNullRows(test_data)
    test_data['CGM Data'] =  test_data['CGM Data'].apply(lambda x: eval(x))
    test_data['Image Before Lunch'] = test_data['Image Before Lunch'].apply(lambda x : np.array(eval(x)))
    test_data = convertTimeStrToTicks(test_data)
    test_data['CGM Data'] = test_data['CGM Data'].apply(lambda x : paddingInsideWithMean(x,300))
    test_data.drop(['Breakfast Time','Lunch Time'],axis = 1,inplace= True)
    cgm_sequences_test = test_data['CGM Data']
    cgm_sequences_test = pad_cgm_sequences(cgm_sequences_test)
    cgm_sequences_test  = pd.DataFrame({'CGM Data': [cgm_sequences_test[i] for i in range(cgm_sequences_test.shape[0])]})
    test_data.drop('CGM Data', axis=1,inplace=True)
    test_data = pd.concat([test_data,cgm_sequences_test], axis=1)
    x_test = test_data.iloc[:,2:]
    y_dummy = np.zeros(x_test.shape[0])
    test_dataset = CGMData(x_test, y_dummy)
    test_dataloader = DataLoader(test_dataset, batch_size=x_test.shape[0], shuffle=False)
    return test_dataloader


if __name__ == '__main__':
    train_dataloader, val_dataloader = read_dataset()
    for cgm_sequence, tabular_features, img_feature ,train_labels in train_dataloader:
        print(f'train_labels : {cgm_sequence}' )
        