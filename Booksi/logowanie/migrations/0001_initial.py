# Generated by Django 5.0.4 on 2024-04-15 16:58

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User_col',
            fields=[
                ('user_id', models.AutoField(primary_key=True, serialize=False)),
                ('location', models.TextField()),
                ('age', models.IntegerField()),
            ],
            options={
                'db_table': 'users',
                'managed': False,
            },
        ),
    ]
