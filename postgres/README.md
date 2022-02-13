# PostgreSQL Cheat Sheet

#### CREATE ROLE
~~~
> CREATE ROLE [role_name];
~~~

pg_roles system catalog
~~~
> SELECT rolname FROM pg_roles;
~~~

 list all existing roles
~~~
> \du
~~~

#### Role attributes `CREATE ROLE [name] WITH [option];`

1. ** Create login roles (place the password in single quotes ('))

~~~
CREATE ROLE alice LOGIN PASSWORD 'securePass1';
~~~

2. Create superuser roles (only superuser create another superuser)

~~~
CREATE ROLE john SUPERUSER LOGIN PASSWORD 'securePass1';
~~~

3. Create roles that can create databases (use `CREATEDB` attribute)

~~~
CREATE ROLE misterdb CREATEDB LOGIN ASSWORD 'Abcd1234';
~~~


#### GRANT (PRIVILEGES) **
After creating a role with the LOGIN attribute, the role cannot do anything, just log in to the PostgreSQL database server.
~~~
GRANT [privilege_list | ALL] ON  [table_name] TO  [role_name];
~~~
* First, specify the privilege_list
  - SELECT, INSERT, UPDATE, DELETE, TRUNCATE, ALL (to grant all privileges)
* Second, specify the name of the table after the ON keyword.
* Third, specify the name of the role to which you want to grant privileges.

~~~
GRANT ALL ON candidates TO joe;
~~~
















































//
