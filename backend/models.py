from django.db import models

# Create your models here.

class Prompt(models.Model):
    name = models.CharField(max_length=100, unique=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

