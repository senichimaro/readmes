# Express Back-End to React
We will use 'react-engine' to render react templates, this allow us render React and passing objects to it.



## react-engine
~~~
npm i react-engine
~~~

##### server.js config
~~~
// # 1. server.js

import express from 'express';
const app = express();


import engine from 'react-engine';
app.engine('.jsx', engine.server.create() );

app.set('view engine', 'jsx');
app.set('views', path.join(__dirname, 'views') );
app.set('view', engine.expressView);


app.get('/', (req, res) => res.render('home'));


app.listen(3000);
~~~






























































//
