import os
import shutil

from main.models import Project
from main.project_type import ProjectType
from deepeye.settings import PROJECT_ROOT, DATASET_DIR_NAME, MODEL_DIR_NAME, DATASET_LIST_DIR_NAME, TMP_DIR_NAME, WEIGHTS_DIR_NAME
from deepeye.settings import DEBUG

if DEBUG:
    PROJECTS = "static"
else:
    PROJECTS = "projects"

# root directory
def create_root_directory():
    os.makedirs(PROJECT_ROOT, exist_ok=True)

# project directory
def get_project_path(project):    
    return os.path.join(PROJECT_ROOT, ProjectType(project.project_type).name, project.name)

def get_project_path_by_project_name(project_name, project_type):
    return os.path.join(PROJECT_ROOT, project_type, project_name)

def create_project_directory(project, delete=False):
    if delete:
        delete_project_directory(project)

    os.makedirs(get_project_path(project), exist_ok=True)

def delete_project_directory(project):
    path = get_project_path(project)
    shutil.rmtree(path, ignore_errors=True)

def delete_project_directory_by_project_name(project_name, project_type):
    path = get_project_path_by_project_name(project_name, project_type)
    shutil.rmtree(path, ignore_errors=True)

# dataset directory
def get_dataset_root(project):
    return os.path.join(get_project_path(project), DATASET_DIR_NAME)

def get_dataset_path(dataset):
    return os.path.join(get_project_path(dataset.project), DATASET_DIR_NAME, dataset.name)

def get_dataset_path_by_dataset_name(dataset_name, project):
    return os.path.join(get_project_path(project), DATASET_DIR_NAME, dataset_name)

def create_dataset_directory(dataset_name, project, delete=False):
    if delete:
        delete_dataset_directory_by_dataset_name(dataset_name, project)

    os.makedirs(get_dataset_path_by_dataset_name(dataset_name, project), exist_ok=True)

def delete_dataset_directory(dataset):
    path = get_dataset_path(dataset)
    shutil.rmtree(pathv, ignore_errors=True)

def delete_dataset_directory_by_dataset_name(dataset_name, project):
    path = get_dataset_path_by_dataset_name(dataset_name, project)
    shutil.rmtree(path, ignore_errors=True)

# model directory
def get_model_path(model):
    return os.path.join(get_project_path(model.project), MODEL_DIR_NAME, model.name)

def get_model_path_by_model_name(model_name, project):
    return os.path.join(get_project_path(project), MODEL_DIR_NAME, model_name)

def create_model_directory_by_model_name(model_name, project, delete=False):
    if delete:
        delete_model_directory_by_model_name(model_name, project)

    os.makedirs(get_model_path_by_model_name(model_name, project), exist_ok=True)

def create_model_directory(model, delete=False):
    if delete:
        delete_model_directory(model)

    os.makedirs(get_model_path(model), exist_ok=True)

def delete_model_directory(model):
    path = get_model_path(model)
    shutil.rmtree(path, ignore_errors=True)

def delete_model_directory_by_model_name(model_name, project):
    path = get_model_path_by_model_name(model_name, project)
    shutil.rmtree(path, ignore_errors=True)

# dataset list directory
def get_dataset_list_directory_path(model_name, project):
    return os.path.join(get_model_path_by_model_name(model_name, project), DATASET_LIST_DIR_NAME)

def create_dataset_list_directory_path(model_name, project, delete=False):
    if delete:
        delete_dataset_list_directory_path(model_name, project)

    os.makedirs(get_dataset_list_directory_path(model_name, project), exist_ok=True)

def delete_dataset_list_directory_path(model_name, project):
    path = get_model_path_by_model_name(model_name, project)
    shutil.rmtree(path, ignore_errors=True)

# weights directory
def get_weights_directory_by_model_name(model_name, project):
    return os.path.join(get_model_path_by_model_name(model_name, project), WEIGHTS_DIR_NAME)

def get_weights_directory(model):
    return os.path.join(get_model_path(model), WEIGHTS_DIR_NAME)

def create_weights_directory_by_model_name(model_name, project, delete=False):
    if delete:
        delete_weights_directory_by_model_name(model_name, project)

    os.makedirs(get_weights_directory_by_model_name(model_name, project), exist_ok=True)

def create_weights_directory(model, delete=False):
    if delete:
        delete_weights_directory(model)
        
    os.makedirs(get_weights_directory(model), exist_ok=True)

def delete_weights_directory_by_model_name(model_name, project):
    path = get_weights_directory_by_model_name(model_name, project)
    shutil.rmtree(path, ignore_errors=True)

def delete_weights_directory(model):
    path = get_weights_directory(model)
    shutil.rmtree(path, ignore_errors=True)

# fine tuning directory
def get_fine_tuning_directory_path_by_model_name(model_name, project):
    return os.path.join(get_model_path_by_model_name(model_name, project), FINE_TUING_DIR_NAME)

def create_fine_tuning_directory_by_model_name(model_name, project, delete=False):
    if delete:
        delete_fine_tuning_directory(model_name, Project)

    os.makedirs(get_fine_tuning_directory_path_by_model_name(model_name, project), exist_ok=True)

def delete_fine_tuning_directory_by_model_name(model_name, project):
    path = get_fine_tuning_directory_path_by_model_name(model_name, project)
    shutil.rmtree(path, ignore_errors=True)

def get_fine_tuning_directory_path(model):
    return os.path.join(get_model_path(model), FINE_TUING_DIR_NAME)

def create_fine_tuning_directory(model, delete=False):
    if delete:
        delete_fine_tuning_directory(model)

    os.makedirs(get_fine_tuning_directory_path(model), exist_ok=True)

def delete_fine_tuning_directory_by_model_name(model_name, project):
    path = get_fine_tuning_directory_path_by_model_name(model_name, project)
    shutil.rmtree(path, ignore_errors=True)

# temporary directory
def get_tmp_directory(project):
    return os.path.join(get_project_path(project), TMP_DIR_NAME)

def create_tmp_directory(project, delete=True):
    if delete:
        delete_tmp_directory(project)

    os.makedirs(get_tmp_directory(project), exist_ok=True)

def delete_tmp_directory(project):
    path = get_tmp_directory(project)
    shutil.rmtree(path, ignore_errors=True)

def get_static_dataset_root_path(project):
    return f'/{PROJECTS}/{ProjectType(project.project_type).name}/{project.name}/{DATASET_DIR_NAME}'

def get_image_uri(project, dataset, img_path):
    return os.path.join(get_static_dataset_root_path(project), dataset.name, img_path)

def get_image_local_full_path(dataset, image_path):
    return os.path.join(PROJECT_ROOT, ProjectType(dataset.project.project_type).name, dataset.project.name, DATASET_DIR_NAME, dataset.name, image_path)

def convert_weights_uri(weight_path):
    return os.path.join(f"/{PROJECTS}", *weight_path.split("/")[4:])

def get_short_name(path, max_len=30, pt="..."):
    if len(path) > max_len:
        length = max_len - len(pt)
        if max_len < 0:
            short_path = path[:max_len]
        else:
            short_path = path[:length] + pt

        return short_path
    else:
        return path

