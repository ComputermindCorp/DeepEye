from django.shortcuts import render, redirect
from django.shortcuts import get_object_or_404
from django.utils import translation
from django.utils.translation import gettext
from .forms import ProjectForm
from .models import Project
from classification.models import Dataset as CL_Dataset
from classification.models import ClassificationModel

from main import file_action
from main.project_type import ProjectType

import os
import shutil

from main.log import get_logger

logger = get_logger(__name__)

def home(request):
    return render(request, 'home.html')

def project_setting(request):
    if request.method == 'POST':
        project_form = ProjectForm(request.POST)
                
        if project_form.is_valid():
            # create database table
            logger.debug("new project making")
            project_name = project_form.cleaned_data['name']
            project_type = ProjectType(project_form.cleaned_data['project_type'])
            project_path = file_action.get_project_path_by_project_name(project_name, project_type.name)

            project = Project(name=project_name, project_path=project_path, project_type=project_type)
            project.save()

            # create directory
            file_action.create_root_directory()
            file_action.create_project_directory(project)
            
            # redirect
            if project_type == ProjectType.object_datection:
                pass
                #return redirect('objectdetection/{}'.format(project.id))
            elif project_type == ProjectType.classification:
                return redirect('classification/{}'.format(project.id))
        else:
            return project_setting_refresh(request, project_form)

    else:
        return project_setting_refresh(request)

def project_setting_refresh(request, project_form=None):
    try:
        project_type = int(request.GET['project_type'])
    except:
        pass

    projects = Project.objects.filter(project_type=project_type).order_by("-update_on")
    logger.debug(projects)
    
    if project_form is None:
        project_form = ProjectForm()

    if project_type == ProjectType.classification:
        title = gettext("Classification")
    else:
        title = ""
    
    return render(request, 'project_setting.html', {
        'title': title,
        'project_type': project_type,
        'projects': projects,
        'n_project': len(projects),
        'project_form': project_form,
    })

def delete_project(request, project_id):
    project = get_object_or_404(Project, id=project_id)    
    project_name = project.name
    project_type = ProjectType(project.project_type)

    if project.project_type == ProjectType.classification:
        models = ClassificationModel.objects.filter(project=project)
    else:
        models = ObjectDetectionModel.objects.filter(project=project)

    models.delete()

    project.delete()
    file_action.delete_project_directory_by_project_name(project_name, project_type.name)

    response = redirect('project_setting')
    response['location'] += '?project_type={}'.format(project_type)
    return response