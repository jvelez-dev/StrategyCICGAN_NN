import pandas as pd
import pymc as pm
import arviz as az
import pytensor
import numpy as np

# Configurar PyTensor para evitar advertencias sobre compiladores C++
pytensor.config.cxx = ""

# ------------------------------
# 1. Cargar y Preprocesar los Datos
# ------------------------------
# Cargar los datos desde el archivo CSV
data = pd.read_csv('synthetic_data.csv')

# Extraer el valor numérico de enemy_attack_time (remover los corchetes)
data['enemy_attack_time'] = data['enemy_attack_time'].str.strip('[]').astype(float)

# Codificar enemy_strategy como 0 para "defense" y 1 para "attack"
data['enemy_strategy'] = data['enemy_strategy'].map({'defense': 0, 'attack': 1})

# Verificar que no haya valores faltantes o inválidos
if data.isnull().values.any():
    print("Advertencia: Se encontraron valores faltantes en los datos. Eliminando filas con NaN...")
    data = data.dropna()

if not np.isfinite(data.values).all():
    print("Advertencia: Se encontraron valores no finitos en los datos. Corrigiendo...")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Revisar las primeras filas para verificar las transformaciones
print("Datos procesados:")
print(data.head())

# ------------------------------
# 2. Definir el Modelo Causal
# ------------------------------
with pm.Model() as causal_model:
    # Priors para los coeficientes
    alpha_marines = pm.Normal('alpha_marines', mu=0, sigma=1)
    beta_marines = pm.Normal('beta_marines', mu=0, sigma=1)
    gamma_marines = pm.Normal('gamma_marines', mu=0, sigma=1)
    delta_marines = pm.Normal('delta_marines', mu=0, sigma=1)  # Nuevo coeficiente
    
    alpha_military = pm.Normal('alpha_military', mu=0, sigma=1)
    beta_military = pm.Normal('beta_military', mu=0, sigma=1)
    gamma_military = pm.Normal('gamma_military', mu=0, sigma=1)
    delta_military = pm.Normal('delta_military', mu=0, sigma=1)  # Nuevo coeficiente
    
    # Variables latentes (predictores observados)
    enemy_minerals = data['enemy_minerals'].values
    enemy_gas = data['enemy_gas'].values
    enemy_strategy = data['enemy_strategy'].values
    enemy_attack_time = data['enemy_attack_time'].values
    
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
        observed=data['enemy_marines']
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
        observed=data['enemy_military_units']
    )
    
    # ------------------------------
    # 3. Inferencia y Guardado del Modelo
    # ------------------------------
    # Muestreo del modelo
    print("Iniciando el muestreo del modelo...")
    causal_trace = pm.sample(500, tune=500, chains=2, random_seed=42)
    
    # Guardar el trace en un archivo NetCDF
    print("Guardando el trace en 'causal_model_trace.nc'...")
    az.to_netcdf(causal_trace, "causal_model_trace.nc")
    
    # Generar predicciones del modelo causal
    with causal_model:
        posterior_predictive = pm.sample_posterior_predictive(causal_trace)

# ------------------------------
# 4. Extraer Predicciones
# ------------------------------
predicted_marines = posterior_predictive.posterior_predictive['enemy_marines'].mean(dim=('chain', 'draw')).values
predicted_military_units = posterior_predictive.posterior_predictive['enemy_military_units'].mean(dim=('chain', 'draw')).values

# Mostrar algunas predicciones
print("Predicciones generadas:")
print("Predicted Marines:", predicted_marines[:5])
print("Predicted Military Units:", predicted_military_units[:5])



