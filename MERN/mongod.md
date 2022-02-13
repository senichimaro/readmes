# `mongod` commands
Use this commandas in Mongo Shell

Select a database to work with
~~~
> use db
~~~

Show Collections
~~~
> show collections
~~~

See into collections (use .pretty() to see formated results)
~~~
> db.userprofiles.find()
> db.userprofiles.find().pretty()
~~~

Remove all documents from the collection
~~~
> db.userprofiles.remove({})
~~~
