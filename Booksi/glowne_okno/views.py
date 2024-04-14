from django.shortcuts import render
from django.http import HttpResponse
from .models import Ksiazki
from django.contrib.auth.decorators import login_required

# Create your views here.
def index(request):
   user_id = request.session.get('user_id')  # Odczyt user_id z sesji
   return render(request, 'glowne_okno/index.html', {'user_id': user_id})

