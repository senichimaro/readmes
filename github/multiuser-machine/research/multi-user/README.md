# Multiple GitHub accounts on the same machine
There are 4 steps
  1. Generating the SSH keys
  2. Adding the new SSH key to the corresponding GitHub account
  3. Registering the new SSH Keys with the ssh-agent
  4. Testing your SSH connection

Check github tutorial for more info: https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/connecting-to-github-with-ssh


## 1. Generating the SSH keys
We can check to see if we have any existing SSH keys: `ls -al ~/.ssh`.

We can generate SSH keys by running:
~~~
# Generating a SSH key
# this could be used for personal purpose
ssh-keygen -t rsa

# Generating a new SSH key with a custom name
ssh-keygen -t rsa -C [email] -f [name]
# example
ssh-keygen -t rsa -C "someone@email.com" -f "funnyName"
~~~


## 2 . Registering the new SSH Keys with the ssh-agent
To use the keys, we have to register them with the ssh-agent on our machine. **Ensure ssh-agent is running using the command `eval "$(ssh-agent -s)"`.**
Add the keys to the ssh-agent like so:
~~~
ssh-add ~/.ssh/id_rsa
ssh-add ~/.ssh/id_rsa_work_user1
~~~


## 3. Adding the new SSH key to the corresponding GitHub account
Copy the public key `clip < ~/.ssh/id_rsa.pub` and then log in to your personal GitHub account:
  1. Go to Settings
  2. Select SSH and GPG keys from the menu to the left.
  3. Click on New SSH key, provide a suitable title, and paste the key in the box below
  4. Click Add key — and you’re done!


## 4. Testing your SSH connection
Enter the following:
~~~
ssh -T git@github.com
~~~

You may see warning or succes message.
  * Warning
~~~
> The authenticity of host 'github.com (IP ADDRESS)' can't be established.
> RSA key fingerprint is SHA256:nThbg6kXUpJWGl7E1IGOCspRomTxdCARLviKw6E5SY8.
> Are you sure you want to continue connecting (yes/no)?
> Verify that the fingerprint in the message you see matches GitHub's RSA public key fingerprint. If it does, then type "yes"
~~~

  * Success
~~~
> Hi username! You've successfully authenticated, but GitHub does not
> provide shell access.
~~~
























































































#
