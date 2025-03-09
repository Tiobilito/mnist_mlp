from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_labels, y_pred_labels))