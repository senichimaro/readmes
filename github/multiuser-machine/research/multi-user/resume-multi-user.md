## Step 1 - Create SSH key pair
Caution: The keys are generated where commands are executed. **It's required to execute this commands in the `kido/.ssh` folder.
~~~
# Generating a SSH key
ssh-keygen -t rsa

$ ssh-keygen -t rsa
Generating public/private rsa key pair.
Enter file in which to save the key (/c/Users/kido/.ssh/id_rsa): senichi
Enter passphrase (empty for no passphrase):
~~~


## Step 2 - Adding SSH key to your GitHub account
1. Goto your GitHub Account -> Settings -> SSH and GPG keys

~~~
C:\Users\kido\.ssh\senichi.pub
~~~

2. Run the following command

~~~
cat id_rsa.pub
~~~

It will prompt something like this:

~~~
Okq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQ34q25erttbb23v34iol2vbip voSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGc uq4248793cm8arñwaerUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8mknlwernlkjewankltjlioipt0943oi325sCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWA
~~~

3. Paste the key in the field
