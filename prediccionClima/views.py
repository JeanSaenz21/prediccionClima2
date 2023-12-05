# views.py en weather_app
import requests_cache
from django.shortcuts import render
from retry_requests import retry
from sklearn.preprocessing import StandardScaler

from .forms import DateRangeForm
import pandas as pd
import tensorflow as tf

from .prediction_model import scaler


def predict_temperature(request):
    if request.method == 'POST':
        form = DateRangeForm(request.POST)
        if form.is_valid():
            date_predict = form.cleaned_data['date_predict']
            # Convertir la fecha proporcionada por el usuario a un formato adecuado
            fecha = pd.to_datetime(date_predict)

            # Crear un DataFrame con la fecha proporcionada y preprocesar los datos
            fecha_data = pd.DataFrame({"temperature_2m_min": [0], "temperature_2m_max": [25], "date": [fecha]})
            fecha_data_scaled = scaler.transform(fecha_data[['temperature_2m_min', 'temperature_2m_max']])

            model = tf.keras.models.load_model("weather_prediction_model.h5")
            # Hacer predicciones en la fecha proporcionada
            prediction = model.predict(fecha_data_scaled)

            return render(request, 'predict_temperature.html', {'predicted_temperature': f"Pronóstico del clima para {fecha.date()}: Temperatura minima {int(prediction[0][0])}°C  Temperatura maxima {int(prediction[0][1])}°C"})
    else:
        form = DateRangeForm()

    return render(request, 'predict_temperature.html', {'form': form})
