from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam, Adadelta

def build_model(optimizer='adam', learning_rate=0.001, hidden_layers=[128, 64]):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(784,)))
    
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
    
    model.add(Dense(10, activation='softmax'))
    
    # Configurar optimizador
    optimizers = {
        'sgd': SGD(learning_rate=learning_rate),
        'adam': Adam(learning_rate=learning_rate),
        'rmsprop': RMSprop(learning_rate=learning_rate),
        'nadam': Nadam(learning_rate=learning_rate),
        'adadelta': Adadelta(learning_rate=learning_rate)
    }
    
    model.compile(
        optimizer=optimizers.get(optimizer.lower(), Adam(learning_rate=learning_rate)),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model