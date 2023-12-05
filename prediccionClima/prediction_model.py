import requests_cache
from retry_requests import retry
from sklearn.preprocessing import StandardScaler
import openmeteo_requests
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error

def get_scaler():
    return scaler

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 31.7202,
    "longitude": -106.4608,
    "daily": ["temperature_2m_max", "temperature_2m_min"],
    "start_date": "2022-01-01",
    "end_date": "2023-12-02"
}
responses = openmeteo.weather_api(url, params=params)

# Procesar la primera ubicación (puedes agregar un bucle para múltiples ubicaciones)
response = responses[0]
print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# Procesar datos diarios
daily = response.Daily()
daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s"),
    end=pd.to_datetime(daily.TimeEnd(), unit="s"),
    freq=pd.Timedelta(seconds=daily.Interval()),
    inclusive="left"
)}
daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["temperature_2m_min"] = daily_temperature_2m_min

daily_dataframe = pd.DataFrame(data=daily_data)

# Preparar los datos para el modelo
X = daily_dataframe[['temperature_2m_min', 'temperature_2m_max']]
y = daily_dataframe[['temperature_2m_min', 'temperature_2m_max']]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=2)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=8, batch_size=32)

# Guardar el modelo entrenado
model.save('weather_prediction_model.h5')

# Análisis de datos básicos con NumPy
temperature_max_np = daily_dataframe['temperature_2m_max'].to_numpy()
temperature_min_np = daily_dataframe['temperature_2m_min'].to_numpy()


# Mapa de calor
correlation_matrix = daily_dataframe.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor - Correlación entre Variables')
plt.show()

# Mapa de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(daily_dataframe['temperature_2m_min'], daily_dataframe['temperature_2m_max'], alpha=0.7)
plt.title('Mapa de Dispersión - Temperatura Mínima vs. Temperatura Máxima')
plt.xlabel('Temperatura Mínima (°C)')
plt.ylabel('Temperatura Máxima (°C)')
plt.show()


mean_max = np.mean(temperature_max_np)
mean_min = np.mean(temperature_min_np)
std_max = np.std(temperature_max_np)
std_min = np.std(temperature_min_np)

# Resumen del análisis de datos
print("Resumen del Análisis de Datos:")
print(f"Media de Temperatura Máxima: {mean_max:.2f}°C")
print(f"Desviación Estándar de Temperatura Máxima: {std_max:.2f}°C")
print(f"Media de Temperatura Mínima: {mean_min:.2f}°C")
print(f"Desviación Estándar de Temperatura Mínima: {std_min:.2f}°C")


