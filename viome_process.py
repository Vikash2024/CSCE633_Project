import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def one_hot(df, col):
    for val in sorted(sorted(df[col].unique()[:])[:-1]):
        df[f'{col}_{val}'] = (df[col] == val) * 1
    return df.drop(col, axis=1)

def process_viome(data):
    data = data.drop(['Subject ID', 'Age'], axis=1)
    data = one_hot(data, 'Race')

    viome_data = pd.DataFrame(zip(*data['Viome'].apply(lambda row: eval(row)).values)).T
    viome_data.columns = [f'Viome_{i}' for i in range(27)]

    data = data.drop(['Viome'], axis=1)
    
    data = pd.concat([data, viome_data], axis=1)
    return data

def load_data():
    viome_data_train = pd.read_csv("demo_viome_train.csv")
    viome_data_test = pd.read_csv("demo_viome_test.csv")

    x_train = process_viome(viome_data_train)
    x_test =  process_viome(viome_data_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test

def plt_optimal_pca(x_train):


    pca = PCA()
    pca.fit(x_train)

    # Explained variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot the cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.savefig("PCA_Check_optimal")

def get_optimal_pca(comps, x_train, x_test):
    pca = PCA(n_components=comps)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    viome_data_train = pd.read_csv("demo_viome_train.csv")
    viome_data_test = pd.read_csv("demo_viome_test.csv")

    x_train.columns = [f'PCA_{i}' for i in range(1,comps + 1)]
    x_test.columns = [f'PCA_{i}' for i in range(1,comps + 1 )]
    x_train = pd.concat([viome_data_train['Subject ID'],x_train],axis = 1)
    x_test = pd.concat([viome_data_test['Subject ID'],x_test],axis = 1)
    return x_train, x_test

def get_processed_pca_csv(x_train, x_test) :
    x_train.to_csv('demo_viome_train_processed.csv')
    x_test.to_csv('demo_viome_test_processed.csv')
    
if __name__ == '__main__':
    components = 27 # Check the Plot to pick the components
    x_train, x_test = load_data()
    plt_optimal_pca(x_train)
    x_train, x_test = get_optimal_pca(components, x_train,x_test)

    get_processed_pca_csv(x_train,x_test)