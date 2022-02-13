

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
