# DeepEye
"DeepEye" is a GUI-based application that can perform DeepLearning.

## Version
0.0.0

## Installation(Dokcer)
### setting
download js and css file.

```
src/
  static/
    css/
      bootstrap.min.css
    js/
      bootstrap.bundle.min.js
      jquery.slim.min.js
      Chart.bundle.min.js
```
  
#### bootstrap.min.css
Ver 4.6.1
https://cdnjs.cloudflare.com/ajax/libs/bootstrap/4.6.1/css/bootstrap.min.css

#### bootstrap.bundle.min.js
ver 4.6.1
https://cdnjs.cloudflare.com/ajax/libs/bootstrap/4.6.1/js/bootstrap.bundle.min.js
#### jquery.slim.min.js
ver 3.5.1
https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.slim.min.js
#### Chart.bundle.min.js
ver 2.9.3
https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.9.3/Chart.bundle.min.js

### build image
```bash
$ cd DeepeEye
$ sudo docker build ./docker/ --add-host 127.0.0.1:0.0.0.0 --tag deepeye
```

### run
```bash
$ sudo docker run --gpus all -it --name deepeye -p 8000:8000 -u $(id -u):$(id -g) -w /deepeye/src -v $PWD:/deepeye deepeye /bin/bash
```

### init
```bash
# docker container
$ python manage.py makemigrations
$ python manage.py migrate
```

### start
```bash
# docker container
$ python manage.py runserver 0.0.0.0:8000
```

# License
This Application is under Apache License 2.0. 
See LICENSE for details