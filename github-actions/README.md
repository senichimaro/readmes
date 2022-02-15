# GitHub Actions
GitHub Actions are a flexible way to automate nearly every aspect of your team's software workflow.

* Automated testing (CI)
* Continuous delivery and deployment
* Responding to workflow triggers using issues, @ mentions, * labels, and more
* Triggering code reviews
* Managing branches
* Triaging issues and pull requests


## Actions and Workflows
A workflow can contain many actions. Each action has its own purpose. We'll put the files relating to the action in their own directories.

### Types of Actions
Actions come in two types: container actions and JavaScript actions.

Docker container actions allow the environment to be packaged with the GitHub Actions code and can only execute in the GitHub-Hosted Linux environment.

JavaScript actions decouple the GitHub Actions code from the environment allowing faster execution but accepting greater dependency management responsibility.


## Welcome to "Hello World" with GitHub Actions
Actions that runs using a workflow file. 

* Step 1: Add a Dockerfile
* Step 2: Add an entrypoint script
* Step 3: Add an Action metadata file
* Step 4: Start your workflow file
* Step 5: Run an Action from your workflow file

#### Most used commands
```
git add . && git commit -m "" && git push origin first-action
```

## Step 1: Add a Dockerfile
Actions will be executed in an environment defined by this file. This action will use a Docker container, so it will require a Dockerfile.

> action-a/Dockerfile

## Step 2: Add an entrypoint script
An entrypoint script must exist in our repository so that Docker has something to execute.

> action-a/entrypoint.sh

In `entrypoint.sh`, we'll outputting a "Hello world" message using an environment variable `$INPUT_MY_NAME` that is called `MY_NAME` in the action file.

## Step 3: Add an Action metadata file
All actions require a config file (called "metadata file") that uses YAML syntax. This file defines the `inputs`, `outputs` and main `entrypoint` for the action.

We will use an input parameter to read in the value of `MY_NAME`.

> action-a/action.yml

## Step 4 & 5: the Workflow file
The workflow file defines an event that fires jobs which steps will run our action.

### Step 4: Start your workflow file
**_Workflows can execute based on your chosen event_**. For this lab, we'll be using the `push` event.

> .github/workflows/main.yml

### Step 5: Run an Action from your workflow file
Workflows piece together jobs, and jobs piece together steps. We create a job that runs an action. 

> .github/workflows/main.yml

### Action file blocks
Actions can be used from within the same repository, from any other public repository, or from a published Docker container image. 

Some important details about why each part of the block exists and what each part does.

* name: A workflow for my Hello World file gives your workflow a name. This name appears on any pull request or in the Actions tab. The name is especially useful when there are multiple workflows in your repository.
* on: push indicates that your workflow will execute anytime code is pushed to your repository, using the push event.
* jobs: is the base component of a workflow run
* build: is the identifier we're attaching to this job
* name: is the name of the job, this is displayed on GitHub when the workflow is running
* runs-on: defines the type of machine to run the job on. The machine can be either a GitHub-hosted runner or a self-hosted runner.
* steps: the linear sequence of operations that make up a job
* uses: actions/checkout@v1 uses a community action called checkout to allow the workflow to access the contents of the repository
* uses: ./action-a provides the relative path to the action we created in the action-a directory of the repository
* with: is used to specify the input variables that will be available to your action in the runtime environment. In this case, the input variable is MY_NAME, and it is currently initialized to "Senichi".

























