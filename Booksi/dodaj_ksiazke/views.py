from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
import pandas as pd
import csv
import json

def get_ISBN():
   books = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\books2.csv')
   unique_isbns_set = set(books['book_id'])
#    print(unique_isbns_set)
   return unique_isbns_set

def dodaj_ksiazke(request):
    isbns = get_ISBN()
    if request.method == 'POST':
        # Odbierz dane z formularza
        user_id = request.session.get('user_id')
        isbn = request.POST.get('ISBN')
        original_title = request.POST.get('Tytuł')
        authors = request.POST.get('Autor')
        original_publication_year = request.POST.get('Rok')
        wydawca = request.POST.get('Wydawca')
        image_url = request.POST.get('Okladka')
        ocena = request.POST.get('Ocena')  # Zmień 'country' na bardziej odpowiedni identyfikator, np. 'ocena'
        puste = ""
        
        print(isbns)
        print(f"UWAGA {ocena}")
        print(isbn)
        
        if isbn not in isbns:
            if original_title != "" and authors != "" and original_publication_year != "" and ocena != "":
                # Zapisz dane do pliku CSV
                with open(r'glowne_okno\program\resources\book_recommendation_dataset\books2.csv', 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([isbn,"","","","","","",authors,original_publication_year,original_title,"","","","","","","","","","","",image_url,""])
                with open(r'glowne_okno\program\resources\book_recommendation_dataset\ratings2.csv', 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([user_id,isbn, ocena])
        else:
            print(f"1UUUUUUUWAGA: {ocena}, {isbn}")
            if ocena != "":
                print(f"UUUUUUUUUUUUUUWAGA: {ocena}, {isbn}")
                # Zapisz dane do pliku CSV
                with open(r'glowne_okno\program\resources\book_recommendation_dataset\ratings2.csv', 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([user_id,isbn,ocena])
            

    context = {
        'isbns': json.dumps(list(isbns)),  # Użyj json.dumps dla bezpiecznego przekazywania listy do JavaScript
    }
    return render(request, 'dodaj_ksiazke/index.html', context)
