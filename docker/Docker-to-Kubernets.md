# Simple steps to Deploy Flask App from Docker to Kubernetes

## CLI & Egnine needed
1. Docker Desktop
2. Kubernetes CLI

### Flask App
1. Flask App
    + app.py
    + requirements.txt

### Docker
2. Dockerfile
```
FROM python:3.7
RUN mkdir /app      # create a dir inside the container
WORKDIR /app/
ADD .               # Copy all into container
RUN pip install -r requirements.txt
CMD ["python", "/app/app.py"]
```
3. Create Docker Image
```
docker build -t [image-tag] .
```

### Kubernetes
- Setup Cluster
  - IAM Role needed
    - EKS
  - Add Node Groups to Cluster
    - IAM Role needed
      - EC2
      - Policy: 
        - `AmazonEKSWorkerNodePolicy`
        - `AmazonEKS_CNI_Policy`
        - `AmazonEC2ContainerRegistryReadOnly`
  - Create kubeconfig (add config to local machine)
    ```
    aws eks --region [region] update-kubeconfig --name [cluster-name]
    kubectl get nodes
    > # nodes data
    ```
- Setting up connectivity to the cluster
  ```
  kubectl create -f https://github.com/[username]/.../pods.yaml
  ```
- Deploy application

1. yaml file `deployment.yaml` (.yaml is kubernetes yaml)
```
# Kubernetes deployment file
# Open deployment.yaml and replace IMAGE_TAG with [DOCKER-USERNAME]/[flask-app].
```
5. depoyment commands
```
kubectl apply -f deployment.yaml
```














