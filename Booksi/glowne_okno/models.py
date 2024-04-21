from django.db import models
# Create your models here.
    
class Book(models.Model):
    ISBN = models.CharField(max_length=20, primary_key=True)
    title = models.CharField(max_length=255, db_column='Book_Title')
    author = models.CharField(max_length=255, db_column='Book_Author')
    year_of_publication = models.CharField(max_length=4, db_column='Year_Of_Publication')
    publisher = models.CharField(max_length=255)
    image_url_s = models.URLField(db_column='Image_URL_S')
    image_url_m = models.URLField(db_column='Image_URL_M')
    image_url_l = models.URLField(db_column='Image_URL_L')
    
    class Meta:
        db_table = 'books'  # Informuje Django, żeby używało istniejącej tabeli
        managed = False

    def __str__(self):
        return f"{self.ISBN} - {self.title} - {self.author}-{self.year_of_publication} - {self.publisher} - {self.image_url_l}"
    
class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    location = models.TextField()
    age = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    class Meta:
        db_table = 'users'
        managed = False

    def __str__(self):
        return f"{self.user_id} - {self.location} - {self.age}"

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_column='User_ID')
    book = models.ForeignKey(Book, on_delete=models.CASCADE, db_column='ISBN')
    rating = models.IntegerField(db_column='Book_Rating')
    class Meta:
        db_table = 'ratings'  # Informuje Django, żeby używało istniejącej tabeli
        managed = False

    def __str__(self):
        return f"{self.user} - {self.book} - {self.rating}"