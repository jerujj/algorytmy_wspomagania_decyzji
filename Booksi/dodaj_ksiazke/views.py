from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
import pandas as pd
import csv
import json

def get_ISBN():
   books = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\Books.csv', usecols=['ISBN', 'Book-Title','Image-URL-L'], dtype={'ISBN': 'str', 'Book-Title': 'str', 'Image-URL-M': 'str'})
   unique_isbns_set = set(books['ISBN'])
   return unique_isbns_set

def dodaj_ksiazke(request):
    isbns = get_ISBN()
    if request.method == 'POST':
        # Odbierz dane z formularza
        user_id = request.session.get('user_id')
        isbn = request.POST.get('ISBN')
        tytul = request.POST.get('Tytuł')
        autor = request.POST.get('Autor')
        rok = request.POST.get('Rok')
        wydawca = request.POST.get('Wydawca')
        link = request.POST.get('Okladka')
        ocena = request.POST.get('Ocena')  # Zmień 'country' na bardziej odpowiedni identyfikator, np. 'ocena'
        puste = ""
        
        if isbn not in isbns:

            if tytul != "" and autor != "" and rok != "" and wydawca != "" and ocena != "":
                # Zapisz dane do pliku CSV
                with open(r'glowne_okno\program\resources\book_recommendation_dataset\Books.csv', 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([isbn, tytul, autor, rok, wydawca,puste,puste,link])
                with open(r'glowne_okno\program\resources\book_recommendation_dataset\Ratings.csv', 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([user_id,isbn, ocena])
        else:
            if ocena != "":
                # Zapisz dane do pliku CSV
                with open(r'glowne_okno\program\resources\book_recommendation_dataset\Ratings.csv', 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([user_id,isbn,ocena])
            

    context = {
        'isbns': json.dumps(list(isbns)),  # Użyj json.dumps dla bezpiecznego przekazywania listy do JavaScript
    }
    return render(request, 'dodaj_ksiazke/index.html', context)
