from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth import get_user_model
from .models import User_col  

def user_login(request):
    if request.method == 'POST':
        user_id = request.POST.get('username') # Zwracana wartość z panelu logowania
        # Sprawdzamy, czy user_id jest liczbą
        if not user_id.isdigit():
            return render(request, 'logowanie/login.html', {'error': 'Numer użytkownika powinien składać się tylko z cyfr.'})
        try:
            user = User_col.objects.get(user_id=user_id)  # Sprawdza czy istnieje użytkownik z podanym user_id
            request.session['user_id'] = user.user_id  # Zapisz user_id do sesji
            return redirect('index')  # Przekierowuje do głównego okna aplikacji
        except User_col.DoesNotExist:
            return render(request, 'logowanie/login.html', {'error': 'Niepoprawny numer użytkownika'})

    return render(request, 'logowanie/login.html')


