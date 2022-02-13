# Flask cheat sheet : in Windows

## simple flask App
~~~
from flask import Flask, jsonify

def create_app(test_config=None):
    # "instance_relative_config"
    # enable relative path within project scoop
    # Here __name__is the name of the current Python module
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/')
    def index():
        return jsonify({"message":"App Running"})

    # Return the app instance
    return app
~~~

1. **Initialize and activate a virtualenv using:**
```
python -m virtualenv env
source env/Scripts/activate
```

2. **Install the dependencies:**
```
pip install -r requirements.txt

# create requirements.txt
# first time
pipreqs

# following time
pipreqs --force
```

3. **Run the development server:** (hot reload)
```
set FLASK_APP=[name_of_base_file]
set FLASK_ENV=development # enables debug mode
flask run
```


































































































































//
