from django.db import models

class Project(models.Model):
    id = models.AutoField("ID", primary_key=True)
    name = models.CharField("Project Name:", default="Project", max_length=255, unique=True, blank=False)
    project_path = models.CharField("Project Path:", default="", max_length=255)
    project_type = models.IntegerField("Project Type:", default=0)
    created_on = models.DateTimeField(auto_now_add=True)
    update_on = models.DateTimeField(auto_now=True)
