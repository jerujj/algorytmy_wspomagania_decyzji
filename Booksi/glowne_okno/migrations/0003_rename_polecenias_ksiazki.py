# Generated by Django 5.0.4 on 2024-04-14 17:37

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('glowne_okno', '0002_rename_polecenia_polecenias'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Polecenias',
            new_name='Ksiazki',
        ),
    ]