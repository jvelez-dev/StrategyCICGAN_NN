# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from google.colab import drive, files
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

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

input_size = 6
model_nn_loaded = NeuralNet(input_size)
model_path = 'modelo_completo.pth'
model_nn_loaded=torch.load(model_path)

model_nn_loaded.eval()

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

def preprocess_data(file_path):
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv(file_path)

    # Extraer el valor numérico de enemy_attack_time (remover los corchetes)
    data['enemy_attack_time'] = data['enemy_attack_time'].str.strip('[]').astype(float)

    # Codificar enemy_strategy como 0 para "defense" y 1 para "attack"
    data['enemy_strategy'] = data['enemy_strategy'].map({'defense': 0, 'attack': 1})

    return data

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

data = preprocess_data('tvt.csv')
causal_model, causal_trace = load_causal_model("causal_model_trace.nc", data)
predicted_marines, predicted_military_units = generate_causal_predictions(causal_model, causal_trace, data)

initial_data_df = pd.read_csv('Eval.csv')
dataf = initial_data_df.iloc[90:100].copy()
#dataf = initial_data_df
bs=[]
bp=[]
i=1
for indice, fila in dataf.iterrows():
    # Configurar el diccionario con los valores de la fila actual
    initial_data = {
        'enemy_minerals': fila['enemy_minerals'],
        'enemy_gas': fila['enemy_gas'],
        'enemy_strategy': fila['enemy_strategy'],
        'enemy_attack_time': fila['enemy_attack_time']
    }
    df_initial = pd.DataFrame(initial_data, index=[0])
    best_strategy, best_probability = generate_optimal_strategy(model_nn_loaded, causal_model, causal_trace, df_initial)
    bs.append(best_strategy)
    bp.append(best_probability)
    print(f"Iteration {i}: Best Strategy: {best_strategy}, Best Probability: {best_probability}")
    i+=1

dataf['best_strategy'] = bs
dataf['best_probability'] = bp
dataf.to_csv('result_eval.csv', mode='a', index=False)

