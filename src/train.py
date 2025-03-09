import numpy as np
from .model import build_model
import json
from datetime import datetime

def train_model(X_train, y_train, X_val, y_val, optimizer='adam', epochs=20, batch_size=64):
    model = build_model(optimizer=optimizer)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )
    return model, history

def save_hyperparameters(config, accuracy):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {
        'timestamp': timestamp,
        'optimizer': config.get('optimizer', 'adam'),
        'learning_rate': config.get('learning_rate', 0.001),
        'hidden_layers': config.get('hidden_layers', [128, 64]),
        'epochs': config.get('epochs', 20),
        'batch_size': config.get('batch_size', 64),
        'val_accuracy': round(accuracy, 4)
    }
    
    # Guardar en archivo JSON
    with open('./results/hyperparams.json', 'a') as f:
        json.dump(record, f)
        f.write('\n')
