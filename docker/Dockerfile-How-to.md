# Use Docker
Raw initial list and steps to work with Docker in Local and push to remote Repo in DockerHub. The DockerHub repo have to be build before push to it (like github).


# Dockerfile
1. create Dockerfile
2. FROM base platform (python/node/etc...) 
3. COPY & WORKDIR current folder
4. RUN installation bash commands (npm i / pip install -r requirements.txt / c...)
5. ENTRYPOINT commands to be ran after `docker run [...]`


# - Commands for Local Development
* `docker build -t try_docker .` : build the image or include changes
* `docker run -dp 5000:5000 try_docker` : run image (match internal config with command target port `[port in docker:localhost port]`)

# - Commands for Push to DockerHub Repo
* `docker login -u USERNAME`
* `docker tag [local-project-name] USERNAME/[remote-project-name]`
* `docker push USERNAME/[remote-project-name]`

# - Docker Commands
* `docker images` : list all available images in local

---

# - Work Flow : from scratch in local
1. Code Ready
2. Create the registry in DockerHub
3. `docker login -u USERNAME`
4. `docker build -t [image-name] .` : create the image
5. `docker tag [image-name] USERNAME/[remote-project-name]` : link the image to the registry
6. `docker push USERNAME/[remote-project-name]` : push the image to the registry

# - Work Flow : pull from Hub
1. `docker login -u USERNAME`
2. get image  `docker pull USERNAME/[remote-project-name]`

# - Work Flow : push to DockerHub
* `docker login -u USERNAME`
* `docker tag [local-project-name] USERNAME/[remote-project-name]`
* `docker push USERNAME/[remote-project-name]`