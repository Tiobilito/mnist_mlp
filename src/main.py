from src.data_preprocessing import load_and_preprocess_data
from src.train import train_model, save_hyperparameters  # Importar ambas funciones
from src.evaluate import evaluate_model
from src.visualize import plot_history, plot_multiple_digits
import json

def main():
    # 1. Preprocesamiento
    X_train, X_val, X_test, y_train, y_val = load_and_preprocess_data()
    
    # 2. Entrenamiento comparativo
    optimizers = ['sgd', 'adam', 'rmsprop', 'nadam', 'adadelta']
    histories = {}
    
    for opt in optimizers:
        print(f"\nEntrenando con {opt.upper()}...")
        model, history = train_model(
            X_train, y_train, 
            X_val, y_val,
            optimizer=opt,
            epochs=20
        )
        histories[opt] = history
        # Registrar hiperparámetros
        save_hyperparameters({
            'optimizer': opt,
            'learning_rate': 0.001,
            'hidden_layers': [128, 64],
            'epochs': 20,
            'batch_size': 64
        }, history.history['val_accuracy'][-1])
    
    # 3. Visualización de resultados
    plot_history(histories, 'accuracy')
    plot_history(histories, 'loss')
    
    # 4. Modelo final con mejor configuración
    final_model, _ = train_model(X_train, y_train, X_val, y_val, optimizer='adam')
    
    # 5. Evaluación detallada
    evaluate_model(final_model, X_val, y_val)
    
    # 6. Predicciones visuales
    y_test_pred = final_model.predict(X_test)
    plot_multiple_digits(X_test, y_test_pred)
    
    # 7. Generar reporte de hiperparámetros
    generate_hyperparam_report()

def generate_hyperparam_report():
    # Leer datos guardados
    with open('./results/hyperparams.json', 'r') as f:
        records = [json.loads(line) for line in f]
    
    # Crear tabla Markdown
    md_table = "| Optimizador | Learning Rate | Capas Ocultas | Épocas | Precisión |\n"
    md_table += "|-------------|---------------|---------------|--------|-----------|\n"
    for record in records:
        md_table += f"| {record['optimizer'].upper()} | {record['learning_rate']} | {record['hidden_layers']} | {record['epochs']} | {record['val_accuracy']*100:.2f}% |\n"
    
    # Guardar en archivo
    with open('/hyperparam_report.md', 'w') as f:
        f.write(md_table)

if __name__ == "__main__":
    main()