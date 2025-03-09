import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data():
    # Cargar datos
    train_data = pd.read_csv('../mnist/train.csv')
    test_data = pd.read_csv('../mnist/test.csv')
    
    # Separar características y etiquetas
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.values
    
    # Normalizar
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # One-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    
    # Dividir en train/val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val