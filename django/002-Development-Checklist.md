# URL Shortner : Second Django Project
Django is a framework that operates from command line prompt with a python environment activated or from conda cmd.

It has initialization commands to create a django project or app and commands to manage a django project to run in the context of the main app manage.py.

## Commands

#### django-admin [command] [params]

* Create Project
~~~
$ django-admin startproject [project-name]
~~~
* Create App
~~~
$ django-admin startapp [name] [directory]
~~~


#### python manage.py [command]
* Run Project
~~~
project-name$ python manage.py runserver
~~~

* Admin: create super user
~~~
$ python manage.py createsuperuser
~~~

* Model: Detect changes in database
~~~
$ python manage.py makemigrations
~~~
* Model: Synchronizes the database
~~~
$python manage.py migrate
~~~


## Basic Project Configs
NOTE: templates and static are configured for a general porpuose or like globally unique. To develop templates and static that live with home route and app routes see movies README.md.

#### templates
* create 'templates' folder in project root
* include the html files

#### static files
* declare STATICFILES_DIRS to BASE_DIR / "static"

#### mainApp/settings.py
* include apps in INSTALLED_APPS
* modify TEMPLATES to DIRS[ BASE_DIR / 'templates']

#### newApp/urls.py
* create urls.py
* from django.urls import path
* from . import views
* declare urlpatterns array
* declare path and view to render

#### newApp/views.py
* define a function associated to a view
* return the rendered file associated to that function

#### mainApp/urls.py
* include path from newApp.urls

#### Use Database
* activation: migrate (first time only)
* createsuperuser (for admin access)
* in newApp/models.py
  + define Schema  
* in newApp/admin.py
  + from .models import ourSchema
  + admin.site.register(ourSchema)
* makemigrations
* migrate




## CRUD (just Create and Read)

## Resume
* Create
  1. declare 'create' path in newApp/urls.py
  2. include {% csrf_token %} within form
  3. set name tags & id submit
  4. declare JavaScript handlers
  5. define 'create' render in newApp/views.py


* Read
  1. define 'read' render in newApp/views.py
    - use shchema objects.get(key=find) method to get data from DB
    - return a redirect()
  2. in newApp/urls.py (a load schema its needed)
    - declare path to read data ('< str : pk >')


## Detailed

* Create
  1. declare 'create' path in newApp/urls.py
    - path('create', views.create, name='create') # no dash ('/')
  2. include {% csrf_token %} within form
  3. set name tags & id submit
  4. declare JavaScript handlers
    - preventDefault
    - declare $.ajax({})
      * type: 'POST'
      * url:'/create'
      * data:{}
        + link: $('#link').val()
        + csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val()
  5. define 'create' render in newApp/views.py
    - imports needed
      * import uuid (id generator)
      * from django.http import HttpResponse
      * from .models import Url
      * from django.utils.html import escape
    - gets the data
      * if request.method == 'POST':
        + url = request.POST['link']
        + escape_url = escape(url)
        + uid = str(uuid.uuid4())[:5]
    - save in db (continue within 'if statement')
      * new_url = Url(link=escape_url, uuid=uid)
      * new_url.save()
    - return HttpResponse(uid)


* Read
  1. in newApp/views.py
    - imports needed
      * from django.shortcuts import path, redirect
      * from django.utils.html import escape
    - define 'read' (request, pk) render
    - use shchema objects.get(key=find) method to get data from DB
    - return a redirect()
  2. in newApp/urls.py (a load schema its needed)
    - declare path to read data ('<str:pk>')





# Deploy in Heroku (from project root)
We will use Heroku as a Hosting Platform and we will creates files at root of the prject.

## Files a& Packages

* install **gunicorn** & **django-heroku** libraries
* create a **runtime.txt** file
* create a **Procfile** file (no extention)
* command prompt configs on git & heroku

#### install gunicorn & django-heroku
~~~
$ pip install gunicorn django-heroku
~~~

#### runtime.txt
This file should containe the python version used in this project.
~~~
$ echo python-3.8.5 >> runtime.txt
~~~

#### Procfile
This file contains commands for Heroku CLI
~~~
web gunicorn urlshortner.wsgi:application --log-file -
~~~

#### requirements.txt
we will use a python library 'pipreqs' (previously installed).

Why not pip freeze? pip freeze only saves the packages that are installed with pip install in your environment, saves packages including those you don't use, and you can't create requirements.txt for a new project without installing modules.

> gunicorn==20.1.0

~~~
$ pipreqs
~~~
> Note: Force or hardcode all needed packeges that not figure into requirements.txt

#### changes in mainApp/settings.py
~~~
# 1. imports
import django_heroku
import dj_database_url

# 2. Production Mode
DEBUG = False

# 3. changes in ALLOWED_HOSTS
ALLOWED_HOSTS = ['.herokuapp.com']
ALLOWED_HOSTS = ['*']

# 4. changes in DATABASES
DATABASES = { 'default': dj_database_url.config() }

# 5. include django_heroku
django_heroku.settings( locals() )
~~~
> ALLOWED_HOSTS = ['*'] works in local

## Heroku actions

#### 1. Login in
~~~
$ heroku login
~~~

#### 2. Create heroku repository
~~~
$ heroku create [repo-name]

# this will return an url.git in the cmd (copy that)
# 'remote git heroku repository'
https://git.heroku.com/my-django-project-name.git
~~~

#### 3. Initialize Git
~~~
$ git init
~~~

#### 4. Relate Git to remote Heroku repo
Paste here the 'remote git heroku repository' (from 2. Create heroku repository)
~~~
$ git remote add heroku [url.git]
~~~

#### 5. Create new Database
~~~
$ heroku addons:create heroku-postgresql:hobby-dev
~~~

#### 6. Git Add, Commit & Push (final step)
~~~
$ git add .
$ git commit -m "CommitMessage"
$ git push heroku master
~~~

#### 7. Migrate
~~~
$ heroku run python manage.py migrate
$ heroku run python manage.py makemigrations
~~~

#### 8. Activate Heroku Ropository
Force at least one instance of the app is running
~~~
$ heroku ps:scale web=1
~~~




# Issues in Development
* DEBUG = False (mainApp/settings.py)

# Issues in Production
* requirements.txt is not updated
* repository migrate db









//
