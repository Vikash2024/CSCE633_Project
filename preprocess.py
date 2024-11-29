# all data preprocessing and dataloader related functions
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.optim.lr_scheduler as lr_scheduler

def convertTimeStrToTicks(data):
    for j in range(len(data['CGM Data'])):
        for i in range(len(data['CGM Data'][j])):
            time_str = data['CGM Data'][j][i][0].split()[1]  # Extract only the time part (HH:MM:SS)
            time_in_ticks = int(datetime.strptime(time_str, "%H:%M:%S").time().hour * 3600 +
                                datetime.strptime(time_str, "%H:%M:%S").time().minute * 60 +
                                datetime.strptime(time_str, "%H:%M:%S").time().second)
            glucose_value = float(data['CGM Data'][j][i][1])
            data['CGM Data'][j][i] = (time_in_ticks, glucose_value)
    return data
def paddingInsideWithMean(data, fixed_increment):
    # Extract the float values and calculate the mean
    float_values = [value for _, value in data]
    mean = sum(float_values) / len(float_values) if data else 0
    
    # Create a list to store the result with missing values filled
    result = []
    last_int = None
    
    for i, (int_val, float_val) in enumerate(data):
        if last_int is None:
            last_int = int_val
            result.append((int_val, float_val))
        else:
            # Check if the current int is a continuation of the expected sequence
            while last_int + fixed_increment < int_val:
                # Insert the missing int with the mean
                last_int += fixed_increment
                result.append((last_int, mean))
            
            # Add the current value
            last_int = int_val      
            result.append((int_val, float_val))

    return result

def removeNullRows(data):
    data = data[data['CGM Data'] != '[]']
    data = data[data['Image Before Lunch'] != '[]']
    data = data.reset_index(drop = True)
    return data
    

def pad_cgm_sequences(cgm_sequences, pad_value=0.0):
    """
    Pads variable-length CGM sequences to the maximum length.
    Args:
        cgm_sequences (list of lists): A list where each item is a sequence of CGM readings.
        pad_value (float): Value to use for padding.
    Returns:
        padded_sequences (torch.Tensor): A tensor of padded sequences.
        lengths (list): Original lengths of the sequences before padding.
    """
    # Convert to tensors and retain original lengths
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in cgm_sequences]
    lengths = [len(seq) for seq in sequences]
    
    # Pad sequences to the maximum length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    
    return padded_sequences.numpy()


if __name__ == '__main__':
    pass