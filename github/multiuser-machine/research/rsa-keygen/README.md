# How to fix - git@github.com permission denied (publickey).
git@github.com: Permission denied (public key).fatal: Could not read from remote repository. - It means GitHub is rejecting your connection because -

1. It is your private repo
1. GitHub does not trust your computer because it does not have the public key of your computer.


~~~
git@github.com: Permission denied (public key).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
~~~



## Step 1 - Create SSH key pair
One of the easiest ways for you to generate a key pair is by running ssh-keygen utility.

Open the command prompt and type in the following
~~~
ssh-keygen
~~~

To keep the ssh-keygen simple, do not enter any key name or passphrase. It will prompt something like this:
~~~
Generating public/private rsa key pair.
Enter file in which to save the key (/Users/rahulwagh/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /Users/rahulwagh/.ssh/id_rsa.
Your public key has been saved in /Users/rahulwagh/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:Okq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWA rahulwagh@local
The key's randomart image is:
+---[RSA 3072]----+
|.ooo..+oo.       |
| oo o..o+.       |
|=E = = +.        |
|*oo X o .        |
|.+ = o  S        |
|o.  + ..         |
|o ..+=+          |
| o + *++         |
|. . o.+.         |
+----[SHA256]-----+
~~~


#### Where to find the key pair
* The file will be generated at - /Users/kido/.ssh/
* Name of the file - id_rsa.pub



## Step 2 - Adding SSH key to your GitHub account
1. Goto your GitHub Account -> Settings
2. Then look for SSH and GPG keys under **Account Settings -> SSH and GPG keys **
3. After that click on New SSH Key. Assign some meaningful name to your key
4. To get the key goto to your command prompt and switch directory path
~~~
C:\Users\rahulwagh.ssh\id_rsa.pub
~~~
5. Run the following command
~~~
cat id_rsa.pub
~~~
It will prompt something like this:
~~~
Okq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQ34q25erttbb23v34iol2vbip voSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGc uq4248793cm8arñwaerUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8mknlwernlkjewankltjlioipt0943oi325sCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWA
~~~
6. Paste the key inside your GitHub account
~~~
# SSH Keys / Add new

. Title
my-new-github-rsakeygen

. Key
Okq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQ34q25erttbb23v34iol2vbip voSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGc uq4248793cm8arñwaerUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWAOkq3w+SesCGLQVToSBQru8mknlwernlkjewankltjlioipt0943oi325sCGLQVToSBQru8RdUZtT2EIIrzH5MQ67DWA

<button>Add SSH Key</button>
~~~
