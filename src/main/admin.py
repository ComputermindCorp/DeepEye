from django.contrib import admin
from main.models import Project
from classification.models import ClassificationModel, Dataset, Result, ClassData, ImageData, DatasetData

# Register your models here.
admin.site.register(Project)

admin.site.register(ClassificationModel)
admin.site.register(Dataset)
admin.site.register(Result)
admin.site.register(ClassData)
admin.site.register(ImageData)
admin.site.register(DatasetData)
