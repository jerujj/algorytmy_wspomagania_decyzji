from django.db import models

class User_col(models.Model):
    user_id = models.AutoField(primary_key=True)
    location = models.TextField()
    age = models.IntegerField()

    class Meta:
        db_table = 'users'  # Informuje Django, żeby używało istniejącej tabeli
        managed = False

    def __str__(self):
        return f"{self.user_id} - {self.location} - {self.age}"
