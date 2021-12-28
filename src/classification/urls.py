from django.urls import path, re_path
from django.views.i18n import JavaScriptCatalog
from django.conf import settings
from django.views.static import serve
from deepeye.settings import DEBUG

from . import views

if DEBUG:
    urlpatterns = [
        path('<int:project_id>', views.classification, name='classification'),
        path('jsi18n/', JavaScriptCatalog.as_view(), name='javascript-catalog'),
        path('delete_dataset/<int:dataset_id>', views.delete_dataset, name='delete_dataset'),
        path('delete_model/<int:model_id>', views.delete_model, name='delete_model'),
        path('dataset_detail/<int:project_id>', views.dataset_detail, name='dataset_detail'),
        path('create_dataset/<int:project_id>', views.create_dataset, name='create_dataset'),
        path('create_model/<int:project_id>', views.create_model, name='create_model'),
        path('training/<int:project_id>', views.training, name='training'),
        path('test_model/<int:project_id>/<int:model_id>', views.test_model, name='test_model'),
        path('test_model_result/<int:project_id>/<int:dataset_id>/<int:test_result_id>', views.test_model_result, name='test_model_result'),
    ]
else:
    urlpatterns = [
        path('<int:project_id>', views.classification, name='classification'),
        path('jsi18n/', JavaScriptCatalog.as_view(), name='javascript-catalog'),
        path('delete_dataset/<int:dataset_id>', views.delete_dataset, name='delete_dataset'),
        path('delete_model/<int:model_id>', views.delete_model, name='delete_model'),
        path('dataset_detail/<int:project_id>', views.dataset_detail, name='dataset_detail'),
        path('create_dataset/<int:project_id>', views.create_dataset, name='create_dataset'),
        path('create_model/<int:project_id>', views.create_model, name='create_model'),
        path('training/<int:project_id>', views.training, name='training'),
        path('test_model/<int:project_id>/<int:model_id>', views.test_model, name='test_model'),
        path('test_model_result/<int:project_id>/<int:dataset_id>/<int:test_result_id>', views.test_model_result, name='test_model_result'),
        re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
        re_path(r'^projects/(?P<path>.*)$', serve,{'document_root': settings.STATIC_PROJECT_ROOT}),
        re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    ]