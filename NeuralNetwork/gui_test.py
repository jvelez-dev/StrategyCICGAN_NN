import tkinter as tk
from tkinter import ttk
import torch
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Importar funciones y clases del archivo NeuralNetwork
# ------------------------------
from NeuralNetwork import (
    load_causal_model,
    generate_causal_predictions,
    preprocess_data,
    NeuralNet,
    generate_optimal_strategy
)

# ------------------------------
# Función para cargar el modelo causal y generar predicciones
# ------------------------------
def load_and_predict(data):
    # Cargar el modelo causal y las predicciones (ajusta las rutas según sea necesario)
    causal_model, causal_trace = load_causal_model("causal_model_trace.nc", data)
    predicted_marines, predicted_military_units = generate_causal_predictions(causal_model, causal_trace, data)
    return causal_model, causal_trace, predicted_marines, predicted_military_units

# ------------------------------
# Función para obtener la estrategia óptima
# ------------------------------
def get_optimal_strategy(input_data):
    # Crear un DataFrame con los datos de entrada
    initial_data = pd.DataFrame([input_data])
    
    # Cargar el modelo causal y generar predicciones causales
    causal_model, causal_trace, predicted_marines, predicted_military_units = load_and_predict(initial_data)
    
    # Crear características combinadas
    X_new = np.column_stack([
        initial_data['enemy_minerals'],
        initial_data['enemy_gas'],
        initial_data['enemy_strategy'],
        initial_data['enemy_attack_time'],
        np.mean(predicted_marines),
        np.mean(predicted_military_units)
    ])
    
    # Escalar los datos (asegúrate de usar el mismo scaler que en el entrenamiento)
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)
    
    # Convertir a tensor
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
    
    # Cargar el modelo neuronal entrenado
    model_nn_loaded = torch.load('modelo_completo.pth')
    model_nn_loaded.eval()
    
    # Generar la estrategia óptima
    best_strategy, best_probability = generate_optimal_strategy(model_nn_loaded, causal_model, causal_trace, initial_data)
    
    return best_strategy, best_probability

# ------------------------------
# Interfaz Gráfica con Tkinter
# ------------------------------
def run_gui():
    def on_submit():
        # Obtener los valores ingresados por el usuario
        enemy_minerals = int(entry_minerals.get())
        enemy_gas = int(entry_gas.get())
        enemy_strategy = strategy_var.get()  # 0 para "defense", 1 para "attack"
        enemy_attack_time = float(entry_attack_time.get())
        
        # Crear un diccionario con los datos de entrada
        input_data = {
            'enemy_minerals': enemy_minerals,
            'enemy_gas': enemy_gas,
            'enemy_strategy': enemy_strategy,
            'enemy_attack_time': enemy_attack_time
        }
        
        # Obtener la estrategia óptima y la probabilidad de victoria
        best_strategy, best_probability = get_optimal_strategy(input_data)
        
        # Mostrar el resultado en la interfaz
        result_label.config(
            text=f"Estrategia Óptima: {best_strategy}\nProbabilidad de Victoria: {best_probability:.4f}"
        )
    
    # Crear la ventana principal
    root = tk.Tk()
    root.title("Recomendador de Estrategias")
    root.geometry("400x350")
    
    # Etiquetas y campos de entrada
    tk.Label(root, text="Recursos Minerales:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    entry_minerals = tk.Entry(root)
    entry_minerals.grid(row=0, column=1, padx=10, pady=5)
    
    tk.Label(root, text="Recursos de Gas:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    entry_gas = tk.Entry(root)
    entry_gas.grid(row=1, column=1, padx=10, pady=5)
    
    tk.Label(root, text="Estrategia:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
    strategy_var = tk.IntVar(value=1)
    ttk.Radiobutton(root, text="Defensa", variable=strategy_var, value=0).grid(row=2, column=1, padx=10, pady=5, sticky="w")
    ttk.Radiobutton(root, text="Ataque", variable=strategy_var, value=1).grid(row=2, column=1, padx=10, pady=5, sticky="e")
    
    tk.Label(root, text="Tiempo de Ataque:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
    entry_attack_time = tk.Entry(root)
    entry_attack_time.grid(row=3, column=1, padx=10, pady=5)
    
    # Botón de envío
    submit_button = tk.Button(root, text="Generar Recomendación", command=on_submit)
    submit_button.grid(row=4, column=0, columnspan=2, pady=10)
    
    # Etiqueta para mostrar el resultado
    global result_label
    result_label = tk.Label(root, text="Resultado aparecerá aquí", font=("Arial", 12), fg="blue", wraplength=380)
    result_label.grid(row=5, column=0, columnspan=2, pady=10)
    
    # Iniciar el bucle principal de la interfaz
    root.mainloop()

# ------------------------------
# Ejecutar la GUI
# ------------------------------
if __name__ == "__main__":
    run_gui()