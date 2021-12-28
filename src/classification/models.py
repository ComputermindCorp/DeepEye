from django.db import models
from django import forms
from main.models import Project
from django.core.validators import MaxValueValidator, MinValueValidator, FileExtensionValidator

class Dataset(models.Model):
    # base param
    id = models.AutoField("ID", primary_key=True)
    name = models.CharField("Dataset Name:", default="Dataset", unique=True, max_length=255, blank=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='classification_dataset_project')
    dataset_path = models.FilePathField("Dataset path:", max_length=255, blank=False)
    created_on = models.DateTimeField(auto_now_add=True)
    classes = models.IntegerField("No. of Classes:", blank=False)
    n_data = models.IntegerField("number of data:", blank=False)

    # temporary percent params
    default_train_ratio = models.IntegerField("Default Train %", default=80, blank=False)
    default_val_ratio = models.IntegerField("Default Validation %", default=10, blank=False)
    default_test_ratio = models.IntegerField("Default Test %", default=10, blank=False)

    memo = models.TextField("Memo", blank=True)

class ClassificationModel(models.Model):
    # base params
    id = models.AutoField("ID", primary_key=True)
    name = models.CharField(max_length=255, default="Model", unique=True, blank=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    created_on = models.DateTimeField(auto_now_add=True)

    # dataset params
    dataset = models.ForeignKey(Dataset, on_delete=models.PROTECT)
    # images
    train_images = models.IntegerField("Train images", default=0, blank=False)
    val_images = models.IntegerField("Val images", default=0, blank=False)
    test_images = models.IntegerField("Test images", default=0, blank=False) 
    # pct
    use_default_ratio = models.BooleanField('Percent Change', default=False)
    train_ratio = models.IntegerField("Train %", default=80, blank=False,
        validators = [
            MaxValueValidator(100),
            MinValueValidator(0),
    ])

    val_ratio = models.IntegerField("Validation %", default=10, blank=False,
        validators = [
            MaxValueValidator(100),
            MinValueValidator(0),
    ])
    test_ratio = models.IntegerField("Test %", default=10, blank=False,
        validators = [
            MaxValueValidator(100),
            MinValueValidator(0),
    ])

    # learning params
    architecture_type = models.CharField("Architecture Type:",
                                         max_length=16,
                                         choices=([('resnet50', 'ResNet50'),
                                                   ('mobilenet', 'MobileNet'),
                                                   ('googlenet', 'GoogleNet'),
                                                   ('vgg16', 'VGG16')]),
                                         default='googlenet')
    epochs = models.IntegerField(default=5,
        validators = [
            MinValueValidator(1),
        ]
    )
    batch_size = models.IntegerField(default=1,
        validators = [
            MinValueValidator(1),
    ])

    learning_rate = models.FloatField(default=0.0001)
    optimizer = models.CharField("Optimizer:",
                             max_length=11,
                             choices=([('sgd', 'sgd'),
                                       ('rmsprop', 'rmsprop'), 
                                       ('adagrad', 'adagrad'),
                                       ('adadelta', 'adadelta'), 
                                       ('adam', 'adam'),
                                       ('adamax', 'adamax'),
                                       ('nadam', 'nadam')]),
                              default='adam')
    
    # fine tuning
    fine_tuning = models.BooleanField("fine tuning", default=False)

    # augmentation param
    image_type = models.CharField("Image Type:",
                                  max_length=9,
                                  choices=([('color', 'Color(3ch)'), 
                                            ('grayscale', 'Grayscale(1ch)')]),
                                  default='color')
    # flip
    horizontal_flip = models.BooleanField("horizontal flip",default=False)
    vertical_flip = models.BooleanField("vertical flip", default=False)

    # rotation
    rotate_30 = models.BooleanField("rotate 30",default=False)
    rotate_45 = models.BooleanField("rotate 45",default=False)
    rotate_90 = models.BooleanField("rotate 90",default=False)

    # processing
    gaussian_noise = models.BooleanField("gaussian noise",default=False)
    blur = models.BooleanField("blur",default=False)
    contrast = models.BooleanField("contrast",default=False)

    # learning result
    epochs_runned = models.IntegerField(default=0)
    train_status = models.TextField(default='none')
    train_loss = models.FloatField(default=0)
    train_acc = models.FloatField(default=0)
    val_loss = models.FloatField(default=0)
    val_acc  = models.FloatField(default=0)
    best_train_loss = models.FloatField(default=0)
    best_val_loss = models.FloatField(default=0)
    best_train_epoch = models.IntegerField(default=0)
    best_val_epoch = models.IntegerField(default=0)

    memo = models.TextField(blank=True)

class Weight(models.Model):
    id = models.AutoField("ID", primary_key=True)
    name = models.CharField(max_length=255, blank=False)
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE)
    path = models.FilePathField(max_length=255, blank=False)

class BaseWeight(models.Model):
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE)
    weight = models.ForeignKey(Weight, on_delete=models.CASCADE)

class Result(models.Model):
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE)
    image = models.CharField(max_length=255)  # result confusion matrix image path
    label = models.CharField(max_length=255)  # TODO class_list txt path
    prd = models.CharField(max_length=255)    # TODO predict result csv file path
    conf = models.CharField(max_length=255)   # 

class ClassData(models.Model):
    id = models.AutoField("ID", primary_key=True)
    class_id = models.IntegerField("class id", blank=False)
    name = models.CharField(max_length=255, blank=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)

class ImageData(models.Model):
    id = models.AutoField("ID", primary_key=True)
    image_id = models.IntegerField("image id", blank=False)
    name = models.CharField(max_length=255, blank=False)
    path = models.FilePathField(max_length=255, blank=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    class_data = models.ForeignKey(ClassData, on_delete=models.CASCADE)

class DatasetData(models.Model):
    id = models.AutoField("ID", primary_key=True)
    data_id = models.IntegerField("data id", blank=False)
    dataset_data_type = models.IntegerField("Dataset Data Type:", default=0)
    image_data = models.ForeignKey(ImageData, on_delete=models.CASCADE)
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE, blank=True, null=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    unique_id = models.IntegerField("temporary unique id", blank=True, null=True)

class TestResult(models.Model):
    id = models.AutoField("ID", primary_key=True)
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE)
    
class Pred(models.Model):
    id = models.AutoField("ID", primary_key=True)
    pred = models.IntegerField()
    test_result = models.ForeignKey(TestResult, on_delete=models.CASCADE)
    image_data = models.ForeignKey(ImageData, on_delete=models.CASCADE)
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE)

class PredProbe(models.Model):
    id = models.AutoField("ID", primary_key=True)
    pred = models.ForeignKey(Pred, on_delete=models.CASCADE)
    value = models.FloatField()

class TrainLog(models.Model):
    id = models.AutoField("ID", primary_key=True)
    epoch = models.IntegerField("epoch")
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE)
    train_loss = models.FloatField(default=0)
    train_acc = models.FloatField(default=0)
    val_loss = models.FloatField(default=0)
    val_acc  = models.FloatField(default=0)