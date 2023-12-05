from django import forms

class DateRangeForm(forms.Form):
    date_predict = forms.DateField(widget=forms.TextInput(attrs={'class': 'form-control', 'type':'date'}))