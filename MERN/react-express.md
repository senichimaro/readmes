# Renderizando React.js en el server con Express.js y react-engine
React.js cada vez es más popular en el mundo del desarrollo web y una de las razones de esto es que permite crear código isomórfico, esto quiere decír que funciona tanto en el cliente (navegador) como en el servidor.

Este artículo usa una versión vieja de React, actualmente este separó las funciones de renderizado a un módulo propio llamado react-dom.

React.js en general es bastante fácil de usar en el servidor, simplemente hay que importar React.js y el componente que queramos renderizar y ejecutar el método de React.js llamado renderToString.

~~~
// # react-render-to-string.js

import React from 'react';
import Home from './views/home';

const html = React.renderToString(<Home />);
~~~

En este ejemplo la constante html es igual a un string con la cual renderizar el componente Home.

Ahora si queremos usar esto en Express.js para renderizar en el servidor tendríamos que hacer algo así:

~~~
// # react-render-express.js

import express from 'express';
import React from 'react';
import Home from './views/home';

const app = express();

app.get('/', (req, res) => {
  const html = React.renderToString(<Home />);

  res.send(html);
});

app.listen(3000);
~~~

De esta forma simplemente estamos enviando el html que generamos como un string usando Express.js.

Ahora esto tiene un problema, y es que no estamos haciendo uso de una función que nos provee Express.js llamada res.render() la cual nos permite usar un sistema de templates para directamente renderizarlo desde Express.js en vez de usar res.send().

Para solucionar esto los ingenieros de PayPal desarrollaron un módulo llamado react-engine. Este módulo nos permite decirle a Express.js que use React.js como engine de templates y nos vuelve a habilitar el uso de res.render() para rendirizar nuestro HTML.

## Introduciendo react-engine
Para hacer uso de react-engine necesitamos bajar este módulo usando el comando:
~~~
npm i -S react-engine
~~~
Una vez descargado lo importamos en nuestro server.js bajo el nombre de engine, luego de iniciar nuestra aplicación de Express.js de la forma tradicional registramos a react-engine como engine de archivos .jsx usando el método engine de nuestra aplicación de Express.js
Luego de esto le decimos a nuestra aplicación donde vamos a tener ubicados nuestras vistas, la extensión de los archivos de nuestras vistas y por último definimos que use react-engine para nuestras vistas.
Luego de esto ya podemos empezar a usar res.render() para renderizar nuestros componentes de React.js como vistas de nuestra aplicación como se ve en el ejemplo:

~~~
// # 1. server.js

import express from 'express';
import engine from 'react-engine';
// iniciamos nuestra aplicación de express
const app = express();
// definimos el engine para archivos jsx
app.engine('.jsx', engine.server.create());
// configuramos la ruta a las vistas
app.set('views', path.join(__dirname, 'views'));
// indicamos que el engine a usar es el de archivos jsx
app.set('view engine', 'jsx');
// le indicamos que use react-engine como engine de nuestras vistas
app.set('view', engine.expressView);

// simplemente llamamos a res.render y react-engine se encarga de renderizar el componente Home para nosotros
app.get('/', (req, res) => res.render('home'));

app.listen(3000);
~~~



~~~
// # 2. home.js

import React from 'react';
import Layout from './layout';

const Home = React.createClass({
  render() {
    return (
      <Layout title="Homepage">
        <h1>Hola mundo</h1>
      </Layout>
    );
  }
});

export default Home;
~~~



~~~
// # 3. layout.js

import React from 'react';

// acá definimos el layout de nuestro HTML donde están los tags html, head, body, etc.
const Layout = React.createClass({
  render() {
    return (
      <html lang="es-AR">
        <head>
          <meta charSet="utf-8" />
          <title>{ this.props.title }</title>
          <meta name="viewport"
            content="width=device-width, initial-scale=1.0" />
          <link rel="stylesheet"
            href="/assets/css/style.css" />
        </head>
        <body>
          { this.props.children }
          <script src="/assets/js/app.js"></script>
        </body>
      </html>
    );
  }
});

export default Layout;
~~~

Esto se puede mejorar incorporando react-router para manejar el ruteo de nuestra aplicación tanto en el servidor como en el cliente, para esto hay que pasarle un objeto con la propiedad reactRoutes, cuyo valor sea la ubicación del archivo donde están las rutas, a la función engine.server.create al momento de definir como funciona el engine de archivos .jsx.

Dentro de nuestro server.js también vamos a definir nuestras rutas y al momento de llamar a res.render le vamos a pasar en vez del nombre de la ruta la propiedad url del objeto req (req.url) lo que va a hacer que react-router sepa cual vista renderizar y como segundo parámetro le pasamos cualquier datos que queramos pasar a nuestra vista (las props que reciba).

Luego necesitamos crear ese archivo de routes que van a estar definidas con react-router y exportarlo. Por último podemos crear un archivo que vamos a usar en el cliente (navegador) para iniciar nuestra aplicación usando react-engine para usar las rutas de react-router.

~~~
// # 1. server.js

import express from 'express';
import engine from 'react-engine';
import path from 'path';

const app = express();
// acá indicamos la ubicación de nuestro archivo con las rutas
app.engine('.jsx', engine.server.create({
  reactRoutes: path.join(__dirname, 'routes.jsx')
}));
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jsx');
app.set('view', engine.expressView);

// definimos nuestra ruta en el servidor, al pasar req.url como primer
// parámetro de res.render react-engine usa react-route para
// renderizar la ruta que corresponda, como segundo parámetro podríamos
// pasar un objeto con los datos (props) que queramos pasar a nuestra vista.
app.get('/', (req, res) => {
  res.render(req.url, {
    title: 'Titulo'
  });
});

app.listen(3000);
~~~



~~~
// # 2. routes.js


import React from 'react';
import Router from 'react-router';
// componentes
import App  from './views/app';
import Home from './views/home';

// configuramos nuestras rutas
const routes = (
  <Router.Route path='/' handler={ App }>
    <Router.DefaultRoute name='home' handler={ Home } />
  </Router.Route>
);

export default routes;
~~~



~~~
// # 3. clients.js


// importamos nuestro archivo de rutas
import routes from './routes.jsx';
// importamos la librería para el navegador de react-engine
import Client from 'react-engine/lib/client';

// por último una vez es listo el DOM iniciamos nuestra aplicación en el navegador
// usando react-engine el cual va a usar react-routes para las rutas en el navegador
document.addEventListener('DOMContentLoaded', function onLoad() {
  // iniciamos el cliente de react-engine pasandole como parámetro un objeto con
  // una propiedad routes igual a las rutas que cargamos de nuestro archivo de rutas
  Client.boot({ routes });
});
~~~



~~~
// # 4. app.js


import React from 'react';
import Layout from './layout.jsx';
import { RouteHandler } from 'react-router';

const App = React.createClass({
  render() {
    return (
      <Layout { ...this.props }>
        <main role="application" className="App" id="app">
          {/* Esta es la parte más importante, acá react-router va inicilizarse
              y a cargar las vistas de cada ruta */}
          <Router.RouteHandler { ...this.props } />
        </main>
      </Layout>
    );
  }
});

export default App;
~~~



~~~
// # 5. home.js


import Header   from '../components/header.jsx';
import React    from 'react';
import { Link } from 'react-router';

const Home = React.createClass({
  render() {
    return (
      <header>
        <h1>{ this.props.title }</h1>
        {/* este link es para que react-router funcione como una SPA */}
        <Link to="home">Home</Link>
      </header>
    );
  }
});

export default Home;
~~~

El último paso sería usar Browserify + Babelify para generar un bundle para el código del cliente y con esto ya tendríamos nuestra aplicación usando React.js para manejar nuestras vistas tanto en el servidor como en el cliente.




















































































































//
