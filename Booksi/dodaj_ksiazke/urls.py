from django.urls import path
from . import views

urlpatterns = [
    path('',views.dodaj_ksiazke,name="dodaj_ksiazke")
]