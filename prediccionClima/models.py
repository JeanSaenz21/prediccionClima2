from django.db import models


class WeatherData(models.Model):
    date = models.DateField()
    max_temperature = models.FloatField()
    min_temperature = models.FloatField()

    def __str__(self):
        return f'{self.date} - Max: {self.max_temperature}, Min: {self.min_temperature}'
