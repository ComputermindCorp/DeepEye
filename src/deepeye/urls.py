"""deepeye URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import include, re_path
from django.conf.urls.i18n import i18n_patterns 
from django.views.i18n import JavaScriptCatalog

from django.contrib import admin
from django.urls import path
from deepeye.settings import DEBUG
from main import views
from django.conf import settings
from django.views.static import serve

if DEBUG:
    urlpatterns = i18n_patterns(
        path('', views.home, name='home'),
        path('project_setting', views.project_setting, name='project_setting'),
        path('delete_project/<int:project_id>', views.delete_project, name='delete_project'),
        path('jsi18n/', JavaScriptCatalog.as_view(), name='javascript-catalog'),
        path('classification/', include('classification.urls')),
        path('admin/', admin.site.urls),
    )
else:
    urlpatterns = i18n_patterns(
        path('', views.home, name='home'),
        path('project_setting', views.project_setting, name='project_setting'),
        path('delete_project/<int:project_id>', views.delete_project, name='delete_project'),
        path('jsi18n/', JavaScriptCatalog.as_view(), name='javascript-catalog'),
        path('classification/', include('classification.urls')),
        path('admin/', admin.site.urls),
        re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
        re_path(r'^projects/(?P<path>.*)$', serve,{'document_root': settings.STATIC_PROJECT_ROOT}),
        re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    )
