# Generated by Django 4.2.7 on 2023-11-12 20:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediccionClima', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='weatherdata',
            name='date',
            field=models.DateField(unique=True),
        ),
    ]