# Movies : First Django Project
This document will guide you through installing Python 3.8 and Django on Windows. See more in [Django Docs](https://docs.djangoproject.com/en/3.2/)

## 1. PC setup
Django is a Python web framework, thus requiring Python to be installed on your machine.

#### Install Python
~~~
py --version
~~~

#### About pip
`pip` is a package manager for Python included by default.

#### Setting up a virtual environment
It is best practice to provide a dedicated environment for each Django project you create. Python itself comes with venv for managing environments which we will use for this guide.

To create a virtual environment for your project, open a new **Anaconda command prompt** and navigate to the folder where you want to create your project, then enter the following:
~~~
py -m venv [project-name]
~~~
This will create a folder called [project-name] if it does not already exist and setup the virtual environment. To activate the environment, in the project folder, inside /Scripts run `activate.bat`:
~~~
activate.bat
~~~
The virtual environment will be activated and you’ll see “(project-name)” next to the command prompt to designate that. Each time you start a new command prompt, you’ll need to activate the environment again.

#### Install Python
Django can be installed easily using pip within your virtual environment.

In the command prompt, ensure your virtual environment is active, and execute the following command:
~~~
py -m pip install Django
~~~
This will download and install the latest Django release in the computer.

After the installation has completed, you can verify your Django installation by executing `django-admin --version` in the command prompt.







## 2. Create Django Project
It is best practice to provide a dedicated environment for each Django project you create.

#### venv
~~~
$ py -m venv [project-name-env]
project-name$ activate.bat
~~~

#### Creating a project
~~~
$ django-admin startproject [project-name]
~~~

The main module will share the name of the project name.

#### Run project
~~~
project-name$ python manage.py runserver
~~~

### Project Structure
The [project structure](https://docs.djangoproject.com/en/2.2/intro/tutorial01/#creating-a-project) files are:

* **The outer `mysite/`** root directory is just a container for your project. Its name **doesn’t matter to Django**; you can rename it to anything you like.

* `manage.py`: A command-line utility that lets you interact with this Django project in various ways. You can read all the details about manage.py in [django-admin and manage.py](https://docs.djangoproject.com/en/2.2/ref/django-admin/).

* **The inner `mysite/`** directory is the actual Python package for your project. **Its name is the Python package name you'll need** to use to import anything inside it (e.g. mysite.urls).

* `mysite/__init__.py`: An empty file that tells Python that this directory should be considered a Python package. If you're a Python beginner, [read more about packages](https://docs.python.org/3/tutorial/modules.html#tut-packages) in the official Python docs.

* `mysite/settings.py`: Settings/configuration for this Django project. [Django settings](https://docs.djangoproject.com/en/2.2/topics/settings/) will tell you all about how settings work.

* `mysite/urls.py`: The URL declarations for this Django project; a table of contents of your Django-powered site. You can read more about URLs in [URL dispatcher](https://docs.djangoproject.com/en/2.2/topics/http/urls/).

* `mysite/wsgi.py`: An entry-point for WSGI-compatible web servers to serve your project. See [How to deploy with WSGI](https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/) for more details.

[WSGI is the Web Server Gateway Interface](https://wsgi.readthedocs.io/en/latest/). It is a specification that describes how a web server communicates with web applications, and how web applications can be chained together to process one request.





## 3. Databases
Django uses SQL database by default, but the same process is to SQL or NoSQL databases initialization. We can instatiate the initial database through `migrate` command.
~~~
$ python manage.py migrate
~~~

###### create super user
Then we can create our superuser to get access to Django built-in Login & Authorization System. After the command execution a config prompt will appear for Username, email (for issues notifications), and Password.
~~~
$ python manage.py createsuperuser
Username:
~~~
**This tutorial set "admin" for Username, no email, and "1234" for password.**






## 4. Recap: Commands Used
Let's recap the commads used til now.


#### 2. Create Django Project
The main module will share the name of the project name.
###### Creating a project
~~~
$ django-admin startproject [project-name]
~~~
###### Run project
~~~
project-name$ python manage.py runserver
~~~


#### 3. Databases
We can instatiate the initial database through `migrate` command.
~~~
$ python manage.py migrate
~~~
###### create super user
~~~
$ python manage.py createsuperuser
~~~


#### On manage.py
~~~
project-name$ python manage.py runserver
~~~
~~~
$ python manage.py migrate
~~~
~~~
$ python manage.py createsuperuser
~~~






## 5. Create our first App
Django is about an apps project, it means Django is a project made by individual app that lives into it. We give it a name to Django at `startproject` in very begining and Django creates the first app for us, the starting app or **the main app**.

Now we will create a new app by commmand that would be secondary or section like. It would have its own modules and different files, like a migration folder related to track the data Schema (the structure of the data).

### django-admin startapp [name] [directory]
Creates a Django app directory structure in the current directory or the given destination.

For example:
~~~
django-admin startapp newApp
~~~
[startapp](https://docs.djangoproject.com/en/3.2/ref/django-admin/) command.



### Project Structure

* `/migrations` folder is where Django stores changes to your database (migrations).

* `__init__.py` tells python that your [name] app is a package.

* `admin.py` is where you register your app's models with the Django admin application.

* `apps.py` is a configuration file common to all Django apps.

* `models.py` is the module containing the models for your app basically ORM modelling.

* `tests.py` contains test procedures which run when testing your app.

* `views.py` is the module containing the views for your app or where we write our business logic.

* `urls.py` it wont be created by default, we have to create this file manually.






## 6. Setting Up a page under newApp/
Now our Django Project is about two apps:  mainApp/ (create by default) & newApp/ (create by startapp command).

Let's add under newApp/ an index view and a child view or section.


###### 1. render the view
A simpler view needs a named function receiving a request parameter that return a HttpResponse (a simple string for now).

In the `newApp/views.py` we have to import HttpResponse from django.http and build our function.
~~~
from django.http import HttpResponse

def index(request):
    return HttpResponse('This is `index` from `newApp/`')
~~~


###### 2. create `newApp/urls.py` file
If we try to navigate to `newApp/` an error will prompt. Its needed a configuration in newApp/urls.py and mainApp/urls.py

**startapp command wont create urls.py by default**, so we have to create this file manually (no cammand, it's just a new file).

The simpler method is to copy everything from `mainApp/urls.py` and adapt it. We need to define the path to the views that be returned.

We have to choose a kind of path configuration, there are three suggested, now we will choose the simpler: function views.

We need to import all views from root to be accesible.
~~~
from django.urls import path
from . import views

urlpatterns = [
    # Function views
    path('', views.index, name='index'),
]
~~~


###### 3. config `mainApp/urls.py`
Here we give visility to created apps by configuring their paths. We need to import path handler and include function from django.urls.
~~~
# default setup
from django.contrib import admin
#
from django.urls import path, include

urlpatterns = [
    # option 3 : Including another URLconf
    path('catalog/', include('catalog.urls')),
    # default setup
    path('admin/', admin.site.urls),
]
~~~


#### child view or section.
Let's create a child view or section under newApp/ (mainApp/newApp/section/).

The process is very similar but the new configuration happens only into newApp/ because a child view or section only need to be configured in their parent (newApp/) not in their grandparent (mainApp/).

So we do it in 2 steps:
* 1. declare the view into `newApp/views.py`
* 2. add the path to the view into `newApp/urls.py`

###### 1. declare the view into `newApp/views.py`
Just declare a new named function.
~~~
def sectionIndex(request):
    return HttpResponse('This `index` from mainApp/newApp/section/')
~~~

###### 2. add the path to the view into `newApp/urls.py`
~~~
urlpatterns = [
# section index
path('section/', views.sectionIndex, name='section'),
# newApp/ index or root index
path('', views.index, name='index'),
]
~~~



## 6. Setting Up a page : Recap
###### 1. render the view
~~~
from django.http import HttpResponse

def index(request):
    return HttpResponse('This is `index` from `newApp/`')
~~~
###### 2. create `newApp/urls.py` file
Import all views from root.
~~~
from django.urls import path
from . import views

urlpatterns = [
    # Function views
    path('', views.index, name='index'),
]
~~~
###### 3. config `mainApp/urls.py`
configuring their paths
~~~
# default setup
from django.contrib import admin
# config paths
from django.urls import path, include

urlpatterns = [
    # option 3 : Including another URLconf
    path('catalog/', include('catalog.urls')),
    # default setup
    path('admin/', admin.site.urls),
]
~~~

#### child view or section.
###### 1. declare the view into `newApp/views.py`
Just declare a new named function.
~~~
def sectionIndex(request):
    return HttpResponse('This `index` from mainApp/newApp/section/')
~~~
###### 2. add the path to the view into `newApp`
~~~
urlpatterns = [
# section index
path('section/', views.sectionIndex, name='section'),
# newApp/ index or root index
path('', views.index, name='index'),
]
~~~





## 7. Templates
It's a good practice to have all our templates inside a folder within it app.

#### 3 Steps
1. create the html into `templates/`
2. modify `views.py`
3. install newApp/ in `settings.py`

###### 1. create the html
Let's create our files into our newly created templates/ folder:
~~~
/newApp
|
|_templates/
  |_index.html
  |_section.html
~~~

###### 2. modify `views.py`
Til now for keep things simple we was returning a string when the route path is requested. Now it's time to render a html file for our viws.

This involves to use the render method with 2 arguments: the request and the file associated to that view.
~~~
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

  def sectionIndex(request):
      return render(request, 'section.html')
~~~

###### 3. install newApp/ in `settings.py`
To render its views our newApp/ needed to be installed into mainApp/. This is made in `settings.py` including it in `INSTALLED_APPS`
~~~
INSTALLED_APPS = [
    #...
    'newApp',
    #...
]
~~~






## 8. Models
Django uses an internal ORM library to manage their Database.

#### 1. The Django ORM
What is an ORM, how does it work, and how should I use one?

[Object-Relational Mapping (ORM)](https://en.wikipedia.org/wiki/Object%E2%80%93relational_mapping) is a technique that lets you query and manipulate data from a database using an object-oriented paradigm.

An ORM library is a completely ordinary library written in your language of choice that encapsulates the code needed to manipulate the data, so you don't use SQL anymore; you interact directly with an object in the same language you're using.

###### Example
You have a book class, you want to retrieve all the books of which the author is "Linus". Manually, you would do something like that:
~~~
book_list = new List();
sql = "SELECT book FROM library WHERE author = 'Linus'";
data = query(sql); // I over simplify ...
while (row = data.next())
{
  book = new Book();
  book.setAuthor(row.get('author');
  book_list.add(book);
}
~~~

With an ORM library, DB calls it would look like this:
~~~
book_list = BookTable.query(author="Linus");
~~~

**The mechanical part is taken care of automatically via the ORM library**.



#### 2. App Model : `models.py`
A model is the single, definitive source of information about your data. It contains the essential fields and behaviors of the data you're storing. Generally, each model maps to a single database table.

###### The basics:
* Each model is a Python class that subclasses django.db.models.Model.

* Each attribute of the model represents a database field.

* With all of this, Django gives you an automatically-generated database-access API; [see Making queries](https://docs.djangoproject.com/en/3.2/topics/db/queries/).

###### Example
This example model defines a Person, which has a first_name and last_name. See [Field Types](https://docs.djangoproject.com/en/3.2/ref/models/fields/#field-types)
~~~
from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
~~~
first_name and last_name are [fields](https://docs.djangoproject.com/en/3.2/topics/db/models/#fields) of the model. Each field is specified as a class attribute, and each attribute maps to a database column.

The above Person model would create a database table like this:
~~~
CREATE TABLE myapp_person (
    "id" serial NOT NULL PRIMARY KEY,
    "first_name" varchar(30) NOT NULL,
    "last_name" varchar(30) NOT NULL
);
~~~
An id field is added automatically, but this behavior can be overridden. See [Automatic primary key fields](https://docs.djangoproject.com/en/3.2/topics/db/models/#automatic-primary-key-fields).

#### 3. Using models
You should think of migrations as a version control system for your database schema. makemigrations is responsible for packaging up your model changes into individual migration files - analogous to commits - and migrate is responsible for applying those to your database. [migrations documentation](https://docs.djangoproject.com/en/3.2/topics/migrations/).

`manage.py migrate` Synchronizes the database state with the current set of models and migrations.

`manage.py makemigrations` Creates new migrations based on the changes detected to your models.



#### 4. Configs in `newApp/admin.py`
To our models be shown in admin it's needed `admin.py` to be configured

* 1. from .models import ourModel
* 2. register ourModel in admin site
~~~
# 1.
from .models import ourModel
# 2.
from django.contrib import admin
admin.site.register(ourModel)
~~~

#### Migration Issues

###### issue : You are trying to add a non-nullable field 'field_name'
Sometimes using simple command python `manage.py makemigrations` get is this error:
~~~
You are trying to add a non-nullable field 'field_name' [etc...]
~~~

###### explanation
It's not possible to add reference to a table that have already data inside

##### Solution
* 1. provide a default value
* 2. declare 'new_field' as a nullable field
* 3. declare and re-declare

###### 1. provide a default value
~~~
comments = models.TextField(default="", max_length=2500)
~~~

###### 2. declare 'new_field' as a nullable field
If you decide to accept 'new_field' as a nullable field you may want to accept 'no input' as valid input for 'new_field'. Then you have to add the blank=True statement as well:
~~~
new_field = models.CharField(blank=True, null=True, max_length=140)
~~~

Even with null=True and/or blank=True you can add a default value if necessary:
~~~
new_field = models.CharField(default='DEFAULT VALUE', blank=True, null=True, max_length=140)
~~~

###### 3. declare and re-declare
* 1. declare
~~~
new_field = models.CharField(max_length=2500, default = "")
~~~
do:
~~~
python manage.py makemigrations
python manage.py migrate
~~~

* 2. re-declare
~~~
# change again:
new_field = models.CharField(max_length=2500)
~~~


###### Field options : null (Field.null)
If True, Django will store empty values as NULL in the database. Default is False.

**Avoid using null on string-based fields such as CharField and TextField**. If a string-based field has null=True, that means it has two possible values for “no data”: NULL, and the empty string. In most cases, it’s redundant to have two possible values for “no data;” the **Django convention is to use the empty string, not NULL**. One exception is when a CharField has both unique=True and blank=True set. In this situation, null=True is required to avoid unique constraint violations when saving multiple objects with blank values.

For both string-based and non-string-based fields, you will also need to set blank=True if you wish to permit empty values in forms, as the null parameter only affects database storage (see blank)

* blank (Field.blank)

If True, the field is allowed to be blank. Default is False.

Note that this is different than null. **null is purely database-related, whereas blank is validation-related**. If a field has blank=True, form validation will allow entry of an empty value. If a field has blank=False, the field will be required.



## 8. Models : Recap
#### 1. The Django ORM (theory)
OOP DB calls it would look like this: (like mangoose)
~~~
book_list = BookTable.find(author="Linus")
~~~

#### 2. App Model : `models.py`
###### Example
This example model defines a Person, which has a first_name and last_name:
~~~
from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
~~~

#### 3. Using models
1. `manage.py makemigrations` Creates new migrations based on the changes detected to your models.

2. `manage.py migrate` Synchronizes the database state with the current set of models and migrations.

#### 4. Configs in `newApp/admin.py`
To our models be shown in admin it's needed `admin.py` to be configured

1. from .models import ourModel
2. register ourModel in admin site

~~~
# 1.
from .models import ourModel
# 2.
from django.contrib import admin
admin.site.register(ourModel)
~~~

###### Migration Issues
**Issue** : You are trying to add a non-nullable field 'field_name'. **Solution** :
1. provide a default value
2. declare 'new_field' as a nullable field
3. declare and re-declare (hack)




## 9. Managing Static Files (e.g. images, JavaScript, CSS)
Files such as images, JavaScript, or CSS. In Django are refer  "static files". Django provides django.contrib.staticfiles to help you manage them. [Django docs](https://docs.djangoproject.com/en/3.2/howto/static-files/)

1. Make sure that `django.contrib.staticfiles` is included in your INSTALLED_APPS in mainApp/settings.py

2. Define `STATIC_URL` in mainApp/settings.py
~~~
STATIC_URL = '/static/'
~~~

3. **Use the template tag in your templates** to build the URL for the given relative path. [static docs](https://docs.djangoproject.com/en/3.2/ref/templates/builtins/#std:templatetag-static)
~~~
{% load static %}
<img src="{% static 'my_app/example.jpg' %}" alt="My image">
~~~

4. Store your **static files in a folder called static** in your app.
~~~
newApp/static/newApp/main.css
# structure
newApp/
    |_static/
      |_newApp/
        |_img/
        |_css/
        |_js/
~~~

###### Deployment
django.contrib.staticfiles provides a convenience management command for gathering static files in a single directory so you can serve them easily.

Run the `collectstatic` management command:
~~~
$ python manage.py collectstatic
~~~

This will copy all files from your static folders into the STATIC_ROOT directory.

Use a web server of your choice to serve the files. Deployment strategies for static files are covered in [Deploying static](https://docs.djangoproject.com/en/3.2/howto/static-files/deployment/).






## 10. Multipage Site
For now we had a site where mainApp/newApp/ and newApp/section/ has pages. Let's keep that and give mainApp/ a page.

It's needed some changes:
1. change newApp/templates/[files] to `newApp/templates/section_name/[files]`
2. change from TEMPLATES[{'DIRS':[]}] to TEMPLATES[{'DIRS':[ BASE_DIR / 'templates']}] in mainApp/`settings.py`
3. create templates folder at project-name root
4. populate project-name/templates/ with page files.
5. create a mainApp/`views.py` and add files to render
6. add STATICFILES_DIRS to mainApp/`settings.py`

###### STATICFILES_DIRS
Your project will probably also have static assets that aren’t tied to a particular app, like in this case, with home view.

STATICFILES_DIRS define in mainApp/`settings.py` file a list of directories (STATICFILES_DIRS) where Django will also look for static files. See [more in docs](https://docs.djangoproject.com/en/3.2/howto/static-files/)
~~~
STATICFILES_DIRS = [
    # relative path
    BASE_DIR / "static",
    # absolut path
    '/var/www/static/',
]
~~~

Now we might be able to get away with **putting our static files directly in mainApp/static/** (rather than creating another mainApp/ subdirectory), but **it would be a bad idea**.

Django will use the first static file it finds whose name matches, and if you had a static file with the same name in a different application, Django would be unable to distinguish between them.

We need to be able to point Django at the right one, and the best way to ensure this is by namespacing them. That is, by putting those static files inside another directory named for the application itself. See [more in docs](https://docs.djangoproject.com/en/3.2/ref/settings/#std:setting-STATICFILES_DIRS)






## 11. Dynamic URLs queries
Let's capture values from the URL.

1. add new path in newApp/urls.py

Let's capture an Integer under newApp/section/
~~~
path('section/<int:catalog_id>/', views.sectionIDcapture, name='section'),
~~~
* <int:  parameter data type
* catalog_id>  custom variable name to the captured value


2. define the render file in newApp/views.py
  1. include the renders
  2. import our database Model
  3. get model data
  4. modify data to a dictionary (best to big objects)
  5. pass data into render

~~~
# 1. include the renders (get_object_or_404 render)
from django.shortcuts import render, get_object_or_404

# 2. import our database Model
from .models import Catalog

def sectionIDcapture(request, catalog_id):
    # 3. get model data
    movie = get_object_or_404(Catalog, pk=catalog_id)

    # 4. modify data to a dictionary
    data = {"movie": movie}

    # 5. pass data into render
    return render(request, 'catalog/section.html', data)

    ## option B : dictionary directly into render
    return render(request, 'newApp/movie_item.html', {"movie": movie})

~~~
_pk is "primary key", here resolves find a record by id_


3. create the html and render data

~~~
<section>

  <h2>{{ movie.title }}</h2>

  <h3>Bio</h3>
  <p>{{ movie.bio }}</p>

  <img src="{{ movie.image }}" />

</section>
~~~













//
