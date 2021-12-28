# django libs
from django.shortcuts import render, redirect
from django.shortcuts import get_object_or_404
from django.core.files.storage import default_storage, FileSystemStorage
from django.core.files.base import ContentFile
from django.contrib import messages
from django.utils.timezone import localtime
from django.utils import translation
from django.utils.translation import gettext
from django.http import HttpResponse, FileResponse
from django.db import models

# deepeye setting & models & form
from deepeye.settings import PROJECT_ROOT, MINIMUM_IMAGES, MODEL_DIR_NAME, FILE_EXT, MAX_CLASS
from main.models import Project
from .forms import DatasetForm, ImportForm, ClassificationForm, TestImportForm, MessageForm
from .models import Dataset, ClassificationModel, ClassData, ImageData, Weight, BaseWeight, TrainLog
from .dataset_util import *

# other libs
import xml.etree.ElementTree as ET
import os
import json
import glob
import csv
import shutil
import random

from main.project_type import ProjectType
from main import file_action
from deepeye.settings import WEIGHTS_DIR_NAME
from .model import DA

from main.log import get_logger

logger = get_logger(__name__)

# ----------------------------------
# front function
# ----------------------------------
def get_datasets(dataset_record):
    dataset_dict = {}
  
    for data in dataset_record:
        dataset_dict.update({data.id: {
            'id': data.id,
            'name': data.name,
            'created_on':str(localtime(data.created_on).strftime('%Y-%m-%d %H:%M:%S')),
            'default_train_ratio':100-(data.default_val_ratio+data.default_test_ratio),
            'default_val_ratio':data.default_val_ratio,
            'default_test_ratio':data.default_test_ratio,
            'classes':data.classes, 
            'n_data':data.n_data,
            'memo':data.memo,}})

    return dataset_record, dataset_dict

def get_models(model_record):
    model_dict = {}

    for data in model_record:
        model_dict.update({data.id: {
            'id': data.id,
            'name': data.name,
            'created_on':str(localtime(data.created_on).strftime('%Y-%m-%d %H:%M:%S')),
            'dataset_name':data.dataset.name,
            'val_ratio':data.val_ratio,
            'test_ratio':data.test_ratio,
            'image_type':data.image_type,
            'architecture_type':data.architecture_type,
            'epochs':data.epochs,
            'batch_size':data.batch_size,
            'learning_rate':data.learning_rate,
            'optimizer':data.optimizer,
            'train_status':data.train_status,
            'epochs_runned':data.epochs_runned,
            'train_loss':data.train_loss,
            'train_acc':data.train_acc, 
            'val_loss':data.val_loss,
            'val_acc':data.val_acc,                                  
            'memo':data.memo,
            'weight_pathes':[{"id":weight.id, "name":weight.name, "path":file_action.convert_weights_uri(weight.path)} for weight in data.weight_set.all()],
        }})

    return model_record, model_dict

def get_train_logs(project):
    models = ClassificationModel.objects.filter(project=project)

    train_logs_dict = {}

    for model in models:
        train_logs = TrainLog.objects.filter(model=model).order_by("epoch")

        data = [[], [], [], []]
        for log in train_logs:
            data[0].append(log.train_loss)
            data[1].append(log.train_acc)
            data[2].append(log.val_loss)
            data[3].append(log.val_acc)

        train_logs_dict[model.id] = {
            'n_epochs': model.epochs,
            'data': data,
            'best_epoch': { 'train': model.best_train_epoch, 'val': model.best_val_epoch },
        }

    return train_logs_dict


def classification(request, project_id):
    logger.info("classification")
    project = get_object_or_404(Project, id=project_id)
    project_type = ProjectType(project.project_type)

    if project_type != ProjectType.classification:
        return redirect("home")

    datasets, dataset_dict = get_datasets(Dataset.objects.filter(project=project))
    models, model_dict = get_models(ClassificationModel.objects.filter(project=project))

    train_logs_dict = get_train_logs(project)

    # model forms
    dataset_form = DatasetForm()
    import_form = ImportForm()
    model_form = ClassificationForm()
    test_form = TestImportForm(prefix='test')
    message_form = MessageForm()

    return render(request, 'classification/classification.html', {
        'project': project,
        'project_id': project.id,
        'project_name': project.name,
        'dataset_form': dataset_form,
        'import_form': import_form,
        'model_form': model_form,
        'test_form': test_form,
        'message_form' : message_form,
        'datasets': datasets,
        'datasets_json': json.dumps(dataset_dict),
        'datasets_len': json.dumps(len(dataset_dict)),
        'models': models,
        'models_json': json.dumps(model_dict),
        'models_len': json.dumps(len(model_dict)),
        'train_logs_json': json.dumps(train_logs_dict),
    })

def delete_dataset(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    dataset_name = dataset.name
    project = dataset.project
    try:
        dataset.delete()
        file_action.delete_dataset_directory_by_dataset_name(dataset_name, project)
    except models.ProtectedError as e:
        messages.error(request, gettext("Data set {} is in use and cannot be deleted.").format(dataset.name))

    return redirect(request.META['HTTP_REFERER'])

def delete_model(request, model_id):
    model = get_object_or_404(ClassificationModel, id=model_id)
    model_name = model.name
    project = model.project

    model.delete()
    file_action.delete_model_directory_by_model_name(model_name, project)

    return redirect(request.META['HTTP_REFERER'])

def dataset_detail(request, project_id):
    project = get_object_or_404(Project, id=project_id)
    
    if 'dataset' in request.GET:
        dataset_id = request.GET['dataset']
        dataset = get_object_or_404(Dataset, id=dataset_id)
    else:
        dataset = None

    if 'model' in request.GET:
        model_id = request.GET['model']
        model = get_object_or_404(ClassificationModel, id=model_id)
        dataset = model.dataset
    else:
        model = None

    if dataset is None:
        raise Http404("No Dataset matches the given query.")

    class_list = load_class_list(dataset)

    image_data_list = ImageData.objects.filter(dataset=dataset)
    n_all_data = len(image_data_list)

    if model is None or model.use_default_ratio:
        train_data_list = DatasetData.objects.filter(dataset=dataset, dataset_data_type=DatasetDataType.Train, model__isnull=True)
        val_data_list = DatasetData.objects.filter(dataset=dataset, dataset_data_type=DatasetDataType.Validation, model__isnull=True)
        test_data_list = DatasetData.objects.filter(dataset=dataset, dataset_data_type=DatasetDataType.Test, model__isnull=True)

        train_ratio = dataset.default_train_ratio
        val_ratio = dataset.default_val_ratio
        test_ratio = dataset.default_test_ratio
    else:
        train_data_list = DatasetData.objects.filter(dataset=dataset, dataset_data_type=DatasetDataType.Train, model=model)
        val_data_list = DatasetData.objects.filter(dataset=dataset, dataset_data_type=DatasetDataType.Validation, model=model)
        test_data_list = DatasetData.objects.filter(dataset=dataset, dataset_data_type=DatasetDataType.Test, model=model)

        train_ratio = model.train_ratio
        val_ratio = model.val_ratio
        test_ratio = model.test_ratio

    n_train_data = len(train_data_list)
    n_val_data = len(val_data_list)
    n_test_data = len(test_data_list)

    train_image_path_list = []
    for data in train_data_list:
        train_image_path_list.append(get_image_uri(data, project))

    val_image_path_list = []
    for data in val_data_list:
        val_image_path_list.append(get_image_uri(data, project))

    test_image_path_list = []
    for data in test_data_list:
        test_image_path_list.append(get_image_uri(data, project))

    return render(request, 'classification/dataset_detail.html', {
        'project': project,
        'dataset': dataset,
        'class_list': class_list,
        'n_all_data': n_all_data,
        'n_train_data': n_train_data,
        'n_val_data': n_val_data,
        'n_test_data': n_test_data,
        'train_data_list': train_data_list,
        'val_data_list': val_data_list,
        'test_data_list': test_data_list,
        'train_image_path_list': train_image_path_list,
        'val_image_path_list': val_image_path_list,
        'test_image_path_list': test_image_path_list,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        })

def create_dataset(request, project_id):
    project = get_object_or_404(Project, id=project_id)
    project_type = ProjectType(project.project_type)

    if project_type != ProjectType.classification:
        return redirect("home")

    if request.method == 'POST':
        dataset_form = DatasetForm(request.POST, request.FILES)
        
        if dataset_form.is_valid():
            dataset_name = request.POST['name']
            default_val_ratio = request.POST['default_val_ratio']
            default_test_ratio = request.POST['default_test_ratio']
            default_train_ratio = 100 - (int(default_val_ratio) + int(default_test_ratio))
            memo = request.POST['memo']
            class_list = set()

            # create dataset directory
            dataset_path = file_action.get_dataset_path_by_dataset_name(dataset_name, project)
            file_action.create_dataset_directory(dataset_name, project, delete=True)
            
            # file upload
            file_field = request.FILES.getlist('file_field')
            dir_list = json.loads(request.POST["dir_list"])

            image_data_list = []

            for filepath, data in zip(dir_list, file_field):
                try:
                    _, class_name, data_name = filepath.split("/")
                    _, ext = os.path.splitext(data_name)
                except:
                    msg = gettext("The image directory should be specified with the following directory structure.") + "\nroot dir/\n__class dir/\n____image file"
                    messages.error(request, msg)
                    return create_dataset_refresh(request, project)

                # file ext check
                if ext in FILE_EXT:
                    class_list.add(class_name)
                    save_path = os.path.join(dataset_path, class_name, data_name)
                    default_storage.save(save_path, ContentFile(data.read()))     
                    image_data_list.append([class_name, data_name])

            class_list = sorted(list(class_list))
            num_classes = len(class_list)

            # create database table
            new_dataset = Dataset(project=project,
                        name=dataset_name,
                        dataset_path=dataset_path,
                        classes=num_classes, 
                        n_data=len(image_data_list),
                        default_train_ratio=default_train_ratio,
                        default_val_ratio=default_val_ratio,
                        default_test_ratio=default_test_ratio,
                        memo=memo)
            new_dataset.save()

            for i, cls in enumerate(class_list):
                new_class_data = ClassData(class_id=i, name=cls, dataset=new_dataset)
                new_class_data.save()

            for i, (class_name, filename) in enumerate(image_data_list):
                cls = ClassData.objects.get(name=class_name, dataset=new_dataset)
                new_image_data = ImageData(image_id=i, name=filename, path=os.path.join(class_name, filename), dataset=new_dataset, class_data=cls)
                new_image_data.save()

            create_image_list(new_dataset, default_val_ratio, default_test_ratio)

            trans_message = gettext("dataset save : success")
            
            messages.success(request, trans_message)

            return redirect('/classification/{}'.format(project.id))
        else:
            return create_dataset_refresh(request, project, dataset_form)
    else:
        return create_dataset_refresh(request, project)
        
def create_dataset_refresh(request, project, dataset_form=None):
    if dataset_form is None:
        dataset_form = DatasetForm()

    return render(request, 'classification/create_dataset.html', {
        'project': project,
        'dataset_form': dataset_form,
    })

def create_model(request, project_id):
    project = get_object_or_404(Project, id=project_id)
    project_type = ProjectType(project.project_type)

    if project_type != ProjectType.classification:
        return redirect("home")

    if request.method == 'POST':
        model_form = ClassificationForm(request.POST)
        
        if model_form.is_valid():
            logger.debug("is_valid()=True")
            dataset = Dataset.objects.get(pk=request.POST['dataset'])
            val_ratio = int(request.POST['val_ratio'])
            test_ratio = int(request.POST['test_ratio'])

            train_ratio = 100 - (val_ratio + test_ratio)
            if train_ratio <=0:
                messages.error(request, gettext("train ratio + val ratio + test ratio = 100%"))
                return create_model_refresh(request, project, model_form)

            use_default_ratio = (dataset.default_val_ratio == val_ratio and dataset.default_test_ratio == test_ratio)

            # params
            model_name = request.POST['name']
            architecture_type = request.POST['architecture_type']
            epochs = int(request.POST['epochs'])
            batch_size = int(request.POST['batch_size'])
            learning_rate = float(request.POST['learning_rate'])
            optimizer = request.POST['optimizer']
            image_type = request.POST['image_type']
            memo = request.POST['memo']

            fine_tuning = request.POST.get('fine_tuning') is not None

            horizontal_flip = 'horizontal_flip' in request.POST
            vertical_flip = 'vertical_flip' in request.POST
            rotate_30 = 'rotate_30' in request.POST
            rotate_45 = 'rotate_45' in request.POST
            rotate_90 = 'rotate_90' in request.POST
            gaussian_noise = 'gaussian_noise' in request.POST
            blur = 'blur' in request.POST
            contrast = 'contrast' in request.POST

            augmentation_flags = {
                'horizontal_flip': horizontal_flip,
                'vertical_flip': vertical_flip,
                'rotate_30': rotate_30,
                'rotate_45': rotate_45,
                'rotate_90': rotate_90,
                'gaussian_noise': gaussian_noise,
                'blur': blur,
                'contrast': contrast,
            }

            # set train, val dataset
            if use_default_ratio:
                image_list_unique_id = None
                train_list, _ = get_dataset_list(project, dataset, DatasetDataType.Train)
                val_list, _ = get_dataset_list(project, dataset, DatasetDataType.Validation)
            else:
                image_list_unique_id = create_image_list(dataset, val_ratio, test_ratio)
                train_list, _ = get_dataset_list(project, dataset, DatasetDataType.Train, image_list_unique_id)
                val_list, _ = get_dataset_list(project, dataset, DatasetDataType.Validation, image_list_unique_id)

            n_iter = len(train_list)
            n_iter = DA.DataAugmentation.get_length(n_iter, augmentation_flags)
            n_iter = n_iter // batch_size

            if n_iter <= 0:
                if not use_default_ratio:
                    DatasetData.objects.filter(unique_id=image_list_unique_id).delete()
                messages.error(request, gettext("train data is not enough.(check batch size and dataset ratio)"))
                return create_model_refresh(request, project, model_form)
            
            logger.debug(f"model_name: {model_name}")
            logger.debug(f"dataset: {dataset.id}:{dataset.name}")
            logger.debug(f"val_ratio: {val_ratio}")
            logger.debug(f"test_ratio: {test_ratio}")
            logger.debug(f"dataset.default_val_ratio: {dataset.default_val_ratio}")
            logger.debug(f"dataset.default_test_ratio: {dataset.default_test_ratio}")
            logger.debug(f"use_default_ratio: {use_default_ratio}")
            logger.debug(f"architecture_type: {architecture_type}")
            logger.debug(f"epochs: {epochs}")
            logger.debug(f"batch_size: {batch_size}")
            logger.debug(f"fine_tuning: {fine_tuning}")

            # fine_tuning check
            if fine_tuning:
                base_model = ClassificationModel.objects.get(pk=request.POST['model'])
                if base_model.architecture_type != architecture_type or base_model.dataset.classes != dataset.classes:
                    messages.error(request, gettext("This model cannot be used with Fine Tuning."))
                    return create_model_refresh(request, project, model_form)
            else:
                base_model = None

            # create model table
            new_model = ClassificationModel(
                name = model_name,
                project = project,
                dataset = dataset,
                use_default_ratio = use_default_ratio,
                train_ratio = train_ratio,
                val_ratio = val_ratio,
                test_ratio = test_ratio,
                architecture_type = architecture_type,
                epochs = epochs,
                batch_size = batch_size,
                learning_rate = learning_rate,
                optimizer = optimizer,
                fine_tuning = fine_tuning,
                image_type = image_type,
                horizontal_flip = horizontal_flip,
                vertical_flip = vertical_flip,
                rotate_30 = rotate_30,
                rotate_45 = rotate_45,
                rotate_90 = rotate_90,
                gaussian_noise = gaussian_noise,
                blur = blur,
                contrast = contrast,
                memo = memo,
            )
            new_model.save()

            # create model directory
            file_action.create_model_directory(new_model, delete=True)

            # weights file
            weights_path = file_action.get_weights_directory(new_model)
            file_action.create_weights_directory(new_model)
            weights_file_path = os.path.join(weights_path, f'{architecture_type}_weights.hdf5')

            new_weight = Weight(
                name = os.path.basename(weights_file_path),
                model = new_model,
                path = weights_file_path,
            )
            new_weight.save()

            # fine tuning
            if fine_tuning:
                base_weight = BaseWeight(model=new_model, weight=base_model.weight_set.get())
                base_weight.save()

            """
            # fine tuning
            weight_file = request.FILES.get('weight_file')
            fine_tuning = weight_file is not None
            if fine_tuning:
                finetuning_weight_path = os.path.join(file_action.get_fine_tuning_directory_path(new_model), weight_file.name)
                create_fine_tuning_directory(new_model)
                default_storage.save(finetuning_weight_path, ContentFile(weight_file.read()))
            """

            # start train
            return render(request, 'classification/training.html', {
                'model_id': new_model.id,
                'model_name':  model_name,
                'dataset_name': dataset.name,
                'project_name':  project.name,
                'dataset':  dataset,
                'use_default_ratio':  use_default_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'architecture_type':  architecture_type,
                'epochs':  epochs,
                'batch_size':  batch_size,
                'learning_rate':  learning_rate,
                'optimizer':  optimizer,
                'fine_tuning':  fine_tuning,
                'image_type':  image_type,
                'horizontal_flip':  horizontal_flip,
                'vertical_flip':  vertical_flip,
                'rotate_30':  rotate_30,
                'rotate_45':  rotate_45,
                'rotate_90':  rotate_90,
                'gaussian_noise':  gaussian_noise,
                'blur':  blur,
                'contrast':  contrast,
                'memo':  memo,
                'weights_path': weights_path,
                'weights_file_path': weights_file_path,
                'image_list_unique_id': image_list_unique_id,
                'n_iter': n_iter,
            })
        else:
            logger.debug("is_valid()=False")
            return create_model_refresh(request, project, model_form)
    else:
        logger.debug("GET")
        return create_model_refresh(request, project)

def create_model_refresh(request, project, model_form=None):
    if model_form is None:
        model_form = ClassificationForm()

    datasets, dataset_dict = get_datasets(Dataset.objects.filter(project=project))
    models, model_dict = get_models(ClassificationModel.objects.filter(project=project))

    return render(request, 'classification/create_model.html', {
        'project': project,
        'model_form': model_form,
        'datasets': datasets,
        'datasets_json': json.dumps(dataset_dict),
        'datasets_len': json.dumps(len(dataset_dict)),
        'models': models,
        'models_json': json.dumps(model_dict),
        'models_len': json.dumps(len(model_dict))
    })

def training(request, project_id):
    project = get_object_or_404(Project, id=project_id)
    project_type = ProjectType(project.project_type)

    return render(request, 'classification/training.html', {
        'project': project,
    })

def test_model(request, project_id, model_id):
    project = get_object_or_404(Project, id=project_id)
    project_type = ProjectType(project.project_type)

    model = get_object_or_404(ClassificationModel, id=model_id)

    datasets, dataset_dict = get_datasets(Dataset.objects.filter(project=project, classes=model.dataset.classes))
    models, model_dict = get_models([model])

    logger.debug(f"model_dict: {model_dict}")

    if project_type != ProjectType.classification:
        return redirect("home")

    return render(request, 'classification/test_model.html', {
        'project': project,
        'project_id': project.id, 
        'project_name': project.name,
        'datasets': datasets,
        'datasets_json': json.dumps(dataset_dict),
        'datasets_len': json.dumps(len(dataset_dict)),
        'model_json': json.dumps(model_dict[model_id]),
        'model_id': model_id,
        'model': model,
        'model_dataset_id': model.dataset.id,
    })


def test_model_result(request, project_id, dataset_id, test_result_id):
    project = get_object_or_404(Project, id=project_id)
    project_type = ProjectType(project.project_type)
    dataset = get_object_or_404(Dataset, id=dataset_id)
    test_result = get_object_or_404(TestResult, id=test_result_id)

    # get predict data
    preds, pred_probs, max_pred_prob, pred_names, labels, label_names, results, image_pathes = get_pred_data(test_result, project)

    # get class names
    class_data_list = ClassData.objects.filter(dataset=dataset).order_by('class_id')
    class_names = [class_data.name for class_data in class_data_list]

    # create confusion matrix
    confusion_matrix, precision, recall, fscore = get_pred_info(test_result, preds, labels)
    pred_info_list = [[c, p, r, f] for c, p, r, f in zip(class_names, precision, recall, fscore)]

    test_result_data = \
        [[pred, pred_name, pred_prob, max_pred_prob, label, label_name, result, path, os.path.basename(path)] \
            for pred, pred_name, pred_prob, max_pred_prob, label, label_name, result, path in zip(preds, pred_names, pred_probs, max_pred_prob, labels, label_names, results, image_pathes)]

    return render(request, 'classification/test_model_result.html', {
        'project': project,
        'project_id': project.id, 
        'project_name': project.name,
        'test_result_id': test_result_id,
        'model': test_result.model,
        'dataset': dataset,
        'test_result_data': test_result_data,
        'class_names': class_names,
        'confusion_matrix': confusion_matrix,
        'pred_info_list': pred_info_list,
    })