from data_preprocessing import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model
from visualize import plot_history, plot_digit
import numpy as np

def main():
    # 1. Preprocesamiento
    X_train, X_val, X_test, y_train, y_val = load_and_preprocess_data()
    
    # 2. Entrenar con diferentes optimizadores
    optimizers = ['sgd', 'adam', 'rmsprop', 'nadam', 'adadelta']
    histories = {}
    
    for opt in optimizers:
        print(f"\nTraining with {opt.upper()}...")
        model, history = train_model(X_train, y_train, X_val, y_val, optimizer=opt)
        histories[opt] = history
        print(f"{opt.upper()} Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    # 3. Visualizar resultados
    plot_history(histories, metric='accuracy')
    plot_history(histories, metric='loss')
    
    # 4. Evaluar modelo final
    final_model, _ = train_model(X_train, y_train, X_val, y_val, optimizer='adam')
    evaluate_model(final_model, X_val, y_val)
    
    # 5. Predecir ejemplo
    y_test_pred = final_model.predict(X_test)
    plot_digit(X_test, y_test_pred[240], index=240)

if __name__ == "__main__":
    main()