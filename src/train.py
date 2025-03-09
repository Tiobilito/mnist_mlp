import numpy as np
from .model import build_model

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