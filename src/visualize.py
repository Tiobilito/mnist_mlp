import matplotlib.pyplot as plt

def plot_history(histories, metric='accuracy'):
    plt.figure(figsize=(12, 5))
    for opt_name, history in histories.items():
        plt.plot(history.history[f'val_{metric}'], label=opt_name.upper())
    plt.title(f'Validation {metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

def plot_digit(X, prediction, index=0):
    plt.imshow(X[index].reshape(28, 28))
    plt.title(f"Predicted: {prediction}")
    plt.axis('off')
    plt.show()