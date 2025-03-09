import matplotlib.pyplot as plt
import numpy as np
import os

# Crear carpeta para resultados si no existe
os.makedirs('./results', exist_ok=True)

def plot_history(histories, metric='accuracy'):
    plt.figure(figsize=(12, 5))
    for opt_name, history in histories.items():
        plt.plot(history.history[f'val_{metric}'], label=opt_name.upper())
    plt.title(f'Validation {metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.savefig(f'./results/val_{metric}.png')  # Guardar gráfica
    plt.close()

def plot_multiple_digits(X, y_pred, y_true=None, num_samples=5):
    indices = np.random.choice(len(X), num_samples, replace=False)
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        title = f"Pred: {np.argmax(y_pred[idx])}"
        if y_true is not None:
            title += f"\nTrue: {np.argmax(y_true[idx])}"
        plt.title(title)
        plt.axis('off')
    plt.savefig('./results/test_samples.png')  # Guardar muestra
    plt.close()