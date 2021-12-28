from django import forms
from django.forms import ModelForm
from .models import Dataset, ClassificationModel
from django.utils.translation import gettext

class DatasetForm(ModelForm):
    class Meta:
        model = Dataset
        fields = ('name', 'default_val_ratio', 'default_test_ratio', 'memo')
        
        widgets = {
            'memo': forms.Textarea(attrs={'rows':7, 'style':"resize: none;" }),
        }
    """
    name = forms.CharField(max_length=255)
    default_val_ratio = forms.IntegerField(min_value=0, max_value=99)
    default_test_ratio = forms.IntegerField(min_value=0, max_value=99)
    memo = forms.CharField(widget=forms.Textarea(attrs={'rows':7, 'style':"resize: none;" }))
    """

    file_field = forms.FileField(widget=forms.ClearableFileInput(
        attrs={'multiple': True, 'webkitdirectory': True, 'directory': True}))


class ImportForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(
        attrs={'multiple': True, 'webkitdirectory': True, 'directory': True}))


class ClassificationForm(ModelForm):
    class Meta: 
        model = ClassificationModel
        fields = ('name',
                  'architecture_type',
                  'epochs',
                  'batch_size',
                  'learning_rate',
                  'optimizer',
                  'use_default_ratio',
                  'val_ratio',
                  'test_ratio',
                  'horizontal_flip',
                  'vertical_flip',
                  'rotate_30',
                  'rotate_45',
                  'rotate_90',
                  'gaussian_noise',
                  'blur',
                  'contrast',
                  'image_type',
                  'memo',
                  'fine_tuning')

        widgets = {
            'memo': forms.Textarea(attrs={'rows':7, 'style':"resize: none;" }),
        }

    #weight_file_field = forms.FileField(required=False)

class TestImportForm(forms.Form):
    model_name = forms.CharField(max_length=100)
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs=
        {'multiple': True}))

class MessageForm(forms.Form):
    message_type = forms.CharField(max_length=100)
    message = forms.CharField(max_length=100)