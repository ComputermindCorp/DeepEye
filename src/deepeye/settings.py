"""
Django settings for deepeye project.

Generated by 'django-admin startproject' using Django 3.1.14.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.1/ref/settings/
"""

import os
from pathlib import Path
from django.contrib.messages import constants as messages
from django.contrib.messages import constants as messages_constants

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '%jd!*wu4@bed+vmm_%ancwi7597%#x^dc-k_8ovt8y0)96#^4('

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ['*']


# Application definition

INSTALLED_APPS = [
    'django_extensions',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main',
    'classification',
    'widget_tweaks',
    'channels',
    'websockets',

    ]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'deepeye.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates'),],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
                "django.contrib.auth.context_processors.auth",
            ],
            'libraries': {
                'custom_templatetags': 'main.templatetags.custom_templatetags'
            }
        },
    },
]

WSGI_APPLICATION = 'deepeye.wsgi.application'

MESSAGE_TAGS ={
    messages.DEBUG:'dark',
    messages.ERROR:'danger',
}

MESSAGE_LEVEL = messages_constants.DEBUG

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/3.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Tokyo'

USE_I18N = True

USE_L10N = True

USE_TZ = True

STATIC_URL = '/static/'
if DEBUG:
    STATICFILES_DIRS = (
        os.path.join(BASE_DIR, 'static'),
        os.path.join(BASE_DIR, 'projects'),
    )
else:
    STATIC_ROOT = os.path.join(BASE_DIR, 'static/')
    STATIC_PROJECT_ROOT = os.path.join(BASE_DIR, 'projects/')

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'projects')

PROJECT_URL = '/projects/'
PROJECT_ROOT = os.path.join(BASE_DIR, 'projects')

DATASET_DIR_NAME = 'datasets'
MODEL_DIR_NAME = 'models'
WEIGHTS_DIR_NAME = 'weights'
FINE_TUING_DIR_NAME = 'finetuning'
TMP_DIR_NAME = "tmp"
DATASET_LIST_DIR_NAME = 'dataset_list'

TEMPLATE_DIRS = [
    os.path.join(os.path.join(BASE_DIR, 'templates')),
]

# Dataset Min/Max Param
MINIMUM_IMAGES = 10
MAX_CLASS = 99

# Channels
ASGI_APPLICATION = 'deepeye.routing.application'

# project
SELECTED_PROJECT = 'test'
SELECTED_MODEL = ''

# image file extension
FILE_EXT = [".jpg", ".jpeg", ".png", ".bmp"]

# localize file
LOCALE_PATHS = ( os.path.join(BASE_DIR, 'locale'), )
