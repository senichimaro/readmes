# How-To deploy React to Heroku
#### The plan 
To have a Production environment - which is Heroku -, a staging environment - which is github master branch -, and a development environment - which are master branches created by todo-tasks. Staging will be updated by PR from it branches.

#### The setup
The project will have two remotes addresses : github [origin] and heroku [heroku].

Heroku uses their own CLI which is installed like a dependency package with `npm i -g heroku`.
1. `heroku login`
2. `heroku create [new-name]` > prompt [website-address] | [heroku-address.git]
3. `git add remote heroku [heroku-address.git]`
4. `git push heroku master`