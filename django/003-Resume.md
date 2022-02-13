# Basic skeleton

1. create './templates' folder
2. create './static' folder
3. modify './settings.py' folder
  * istall newApp
  * templates config : [ BASE_DIR / 'templates']
  * static files : STATICFILES_DIRS to BASE_DIR / 'static'
4. create 'newApp/urls.py'
  * create urls.py
5. define newApp/views.py
  * define a function associated to a view
6. include mainApp/urls.py
  * include path from newApp.urls


# CRUD logic

## Create
1. set Schema (newApp/models.py)
2. register Schema (newApp/admin.py)
3. set a form
4. get data from the view
5. import the Schema in the view
6. insert data into object Schema within a variable
7. save the newly object Schema insertion

## Read
8. get all DB object records
9. include the object records in the response
10. return user redirect to his own page
11. include a loop in the view to display object records

## Delete
12. set a new url to get data sent
13. send form data to that url
14. create a new view
15. capture in the view data sent (object id)
16. use the id to find its object record
17. delete it
16. return redirect
15. set a link pointing to that url carraing the object id clicked



//
