from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd

# Cargar los datos reales
df = pd.read_csv("tvt.csv")

# Definir la metadata del dataset
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
# Revisar y ajustar tipos de datos si es necesario
metadata.update_column(column_name="enemy_strategy", sdtype="categorical")
metadata.update_column(column_name="enemy_marines", sdtype="numerical")
# Crear el sintetizador con CTGAN
synthesizer = CTGANSynthesizer(
    metadata,
    epochs=3000,  # Más iteraciones para mejorar la calidad
    batch_size=512,  # Tamaño del batch mayor para estabilizar entrenamiento
    generator_lr=2e-4,  # Ajustar learning rate del generador
    discriminator_lr=2e-4,  # Ajustar learning rate del discriminador
    embedding_dim=256,  # Dimensión del embedding (aumentarlo puede ayudar)
    generator_decay=1e-6,  # Decaimiento de peso para evitar overfitting
    pac = 1 # Asegura que el batch sea múltiplo de packet_size
)


# Entrenar el modelo
synthesizer.fit(df)

# Generar 5000 registros sintéticos
synthetic_data = synthesizer.sample(5000)


# Guardar los datos sintéticos
synthetic_data.to_csv("synthetic_data.csv", index=False)

print("✅ Datos sintéticos generados exitosamente")