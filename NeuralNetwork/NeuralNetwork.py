import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# ------------------------------
# 1. Función para Cargar el Modelo Causal
# ------------------------------
def load_causal_model(trace_path, data):
    # Cargar el trace del modelo causal previamente guardado
    causal_trace = az.from_netcdf(trace_path)
    
    # Reconstruir el modelo causal
    with pm.Model() as causal_model:
        # Priors para los coeficientes
        alpha_marines = pm.Normal('alpha_marines', mu=0, sigma=1)
        beta_marines = pm.Normal('beta_marines', mu=0, sigma=1)
        gamma_marines = pm.Normal('gamma_marines', mu=0, sigma=1)
        delta_marines = pm.Normal('delta_marines', mu=0, sigma=1)
        
        alpha_military = pm.Normal('alpha_military', mu=0, sigma=1)
        beta_military = pm.Normal('beta_military', mu=0, sigma=1)
        gamma_military = pm.Normal('gamma_military', mu=0, sigma=1)
        delta_military = pm.Normal('delta_military', mu=0, sigma=1)
        
        # Variables latentes (predictores observados)
        enemy_minerals = pm.Data('enemy_minerals', [])
        enemy_gas = pm.Data('enemy_gas', [])
        enemy_strategy = pm.Data('enemy_strategy', [])
        enemy_attack_time = pm.Data('enemy_attack_time', [])
        
        # Relaciones
        enemy_marines_est = (
            alpha_marines * enemy_minerals +
            beta_marines * enemy_gas +
            gamma_marines * enemy_strategy +
            delta_marines * enemy_attack_time
        )
        enemy_marines_obs = pm.Normal(
            'enemy_marines',
            mu=enemy_marines_est,
            sigma=pm.HalfNormal('sigma_marines', sigma=1),
            observed=data['enemy_marines'].values
        )
        
        enemy_military_units_est = (
            alpha_military * enemy_minerals +
            beta_military * enemy_gas +
            gamma_military * enemy_strategy +
            delta_military * enemy_attack_time
        )
        enemy_military_units_obs = pm.Normal(
            'enemy_military_units',
            mu=enemy_military_units_est,
            sigma=pm.HalfNormal('sigma_military', sigma=1),
            observed=data['enemy_military_units'].values
        )

    return causal_model, causal_trace

# ------------------------------
# 2. Función para Generar Predicciones del Modelo Causal
# ------------------------------
def generate_causal_predictions(model, trace, data):
    with model:
        # Actualizar los valores de las variables observadas
        pm.set_data({
            'enemy_minerals': data['enemy_minerals'].values,
            'enemy_gas': data['enemy_gas'].values,
            'enemy_strategy': data['enemy_strategy'].values,
            'enemy_attack_time': data['enemy_attack_time'].values
        })
        # Generar predicciones
        posterior_predictive = pm.sample_posterior_predictive(trace)
    
    predicted_marines = posterior_predictive.posterior_predictive['enemy_marines'].mean(dim=('chain', 'draw')).values
    predicted_military_units = posterior_predictive.posterior_predictive['enemy_military_units'].mean(dim=('chain', 'draw')).values
    
    return predicted_marines, predicted_military_units

# ------------------------------
# 3. Preprocesamiento de Datos
# ------------------------------
def preprocess_data(file_path):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(file_path)
    
    # Extraer el valor numérico de enemy_attack_time (remover los corchetes)
    data['enemy_attack_time'] = data['enemy_attack_time'].str.strip('[]').astype(float)
    
    # Codificar enemy_strategy como 0 para "defense" y 1 para "attack"
    data['enemy_strategy'] = data['enemy_strategy'].map({'defense': 0, 'attack': 1})
    
    return data

# ------------------------------
# 4. Definir y Entrenar la Red Neuronal
# ------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        #self.dropout1 = nn.Dropout(0.3)  # Dropout del 30%
        self.fc2 = nn.Linear(128, 64)
        #self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #x = self.dropout2(x)
        x = self.sigmoid(self.fc4(x))
        return x

def train_neural_network_with_plot(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    model_nn = NeuralNet(input_size=X_train_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)
    
    train_accuracies = []
    test_accuracies = []
    losses = []
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    def calculate_accuracy(model, X, y):
        with torch.no_grad():
            predictions = model(X)
            predicted_classes = (predictions > 0.5).float()
            accuracy = (predicted_classes == y).float().mean()
        return accuracy.item()
    
    for epoch in range(1000):
        # Forward pass
        outputs = model_nn(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        losses.append(loss.item())
        
        # Backward pass y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #scheduler.step()

        # Calcular precisiones
        train_accuracy = calculate_accuracy(model_nn, X_train_tensor, y_train_tensor)
        test_accuracy = calculate_accuracy(model_nn, X_test_tensor, y_test_tensor)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    
    # Graficar la curva de aprendizaje
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
    
    return model_nn

# ------------------------------
# 5. Generar Estrategias Óptimas
# ------------------------------
def generate_optimal_strategy(model_nn, causal_model, causal_trace, initial_data):
    best_probability = 0
    best_strategy = None
    
    # Probar diferentes combinaciones de enemy_strategy y enemy_attack_time
    for strategy in [0, 1]:  # 0 = defense, 1 = attack
        for attack_time in np.linspace(5, 20, 10):  # Valores de tiempo de ataque entre 5 y 20
            new_data = initial_data.copy()
            new_data['enemy_strategy'] = [strategy]
            new_data['enemy_attack_time'] = [attack_time]
            
            # Generar predicciones del modelo causal
            predicted_marines_new, predicted_military_units_new = generate_causal_predictions(causal_model, causal_trace, new_data)
            
            # Tomar el valor medio de las predicciones causales
            predicted_marines_mean = np.mean(predicted_marines_new)
            predicted_military_units_mean = np.mean(predicted_military_units_new)
            
            # Crear un tensor con los nuevos datos
            X_new = np.column_stack([
                new_data['enemy_minerals'],
                new_data['enemy_gas'],
                new_data['enemy_strategy'],
                new_data['enemy_attack_time'],
                predicted_marines_mean,
                predicted_military_units_mean
            ])
            X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
            
            # Predecir el resultado con la red neuronal
            with torch.no_grad():
                prediction = model_nn(X_new_tensor)
                probability = prediction.item()
            
            # Guardar la mejor estrategia
            if probability > best_probability:
                best_probability = probability
                best_strategy = {'strategy': strategy, 'attack_time': attack_time}
    
    return best_strategy, best_probability

# ------------------------------
# 6. Ejecución Principal
# ------------------------------
if __name__ == "__main__":
    # Preprocesar los datos
    data = preprocess_data('synthetic_data.csv')
    
    # Cargar el modelo causal
    causal_model, causal_trace = load_causal_model("causal_model_trace.nc", data)
    
    # Generar predicciones del modelo causal
    predicted_marines, predicted_military_units = generate_causal_predictions(causal_model, causal_trace, data)
    
    # Crear un conjunto de datos combinado
    X = np.column_stack([
        data['enemy_minerals'], 
        data['enemy_gas'], 
        data['enemy_strategy'], 
        data['enemy_attack_time'], 
        predicted_marines, 
        predicted_military_units
    ])
    y = data['enemy_result'].values
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convertir a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Entrenar la red neuronal
    model_nn = train_neural_network_with_plot(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    
    # Guardar el modelo entrenado
    #torch.save(model_nn.state_dict(), 'model_nn.pth')
    
    # Opcional: Cargar el modelo guardado
    #model_nn_loaded = NeuralNet(input_size=X_train_tensor.shape[1])
    #model_nn_loaded.load_state_dict(torch.load('model_nn.pth'))
    #model_nn_loaded.eval()
    model_nn.eval()
    # Generar una estrategia óptima
    #initial_data = {
    #    'enemy_minerals': [500],
    #    'enemy_gas': [300],
    #    'enemy_strategy': [1],  # Inicialmente "attack"
    #    'enemy_attack_time': [15.0]
    #}
    # Generar 100 elementos aleatorios para cada variable
    np.random.seed(42)  # Fijar la semilla para reproducibilidad

    initial_data = {
        'enemy_minerals': np.random.randint(0, 1000, size=100).tolist(),  # Valores entre 0 y 1000
        'enemy_gas': np.random.randint(0, 500, size=100).tolist(),         # Valores entre 0 y 500
        'enemy_strategy': np.random.choice([0, 1], size=100).tolist(),     # 0 = defense, 1 = attack
        'enemy_attack_time': np.random.uniform(5, 20, size=100).tolist()   # Valores entre 5 y 20
    }
    initial_data_df = pd.DataFrame(initial_data)
    #best_strategy, best_probability = generate_optimal_strategy(model_nn, causal_model, causal_trace, initial_data_df)
    
    #print(f"Estrategia óptima: {best_strategy}")
    #print(f"Probabilidad de victoria: {best_probability:.4f}")
    """best_strategies = []
    best_probabilities = []
    
    for _, row in initial_data_df.iterrows():
        input_data = row.to_dict()
        best_strategy, best_probability = generate_optimal_strategy(model_nn, causal_model, causal_trace, pd.DataFrame([input_data]))
        best_strategies.append(best_strategy)
        best_probabilities.append(best_probability)
    
    # Agregar las mejores estrategias y probabilidades al DataFrame
    initial_data_df['best_strategy'] = best_strategies
    initial_data_df['best_probability'] = best_probabilities
    
    # Mostrar los resultados
    print(initial_data_df[['enemy_minerals', 'enemy_gas', 'enemy_strategy', 'enemy_attack_time', 'best_strategy', 'best_probability']].head())"""
    initial_data_df.to_csv("Eval.csv", index=False, header=True)

    torch.save(model_nn, 'modelo_completo.pth')