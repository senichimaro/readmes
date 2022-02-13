# Multiple GitHub accounts on the same machine (ssh config)

## 3 Steps
1. rsa key   (one per user)
2. Global `config` file in `~/.ssh/`
3. Local `config` within the repo (in `./project-name/.git/config`)

#### 1. rsa key   (one per user)

~~~
# Generating a SSH key
ssh-keygen -t rsa -C "any@email.com" -f "anyName"

# enable ssh-agent
eval "$(ssh-agent -s)"

# Register with ssh-agent the new SSH Keys
ssh-add ~/.ssh/anyName

# Adding SSH key to GitHub -> Settings -> SSH and GPG keys (copy/paste)
# key inside this [file].pub filename
# 'cat' command will prompt it out to copy/paste
cat anyName.pub
~~~

#### 2. Global `config` file in `~/.ssh/`

It's possible to have several [aliases] for the same hostname and create a reference to diferents users with it own authorization. We can configure ssh to use a specific encryption key depending on the host (Host alias).

Create a `config` file in `~/.ssh/` :
~~~
# Default GitHub
Host [alias1]
  HostName github.com
  User git
  IdentityFile ~/.ssh/alias1

# Professional github alias
Host [alias2]
  HostName github.com
  User git
  IdentityFile ~/.ssh/alias2
~~~

#### 3. Local `config` within the local repo (in `./project-name/.git/config`)


A `./project-name/.git/config` will be more or less like this example:
~~~
[core]
    [...]
[remote "origin"]
	url = git@github.com:user2/web-project.git
	[...]

~~~

This is the target line: (this line is the remote)
~~~
url = git@github.com:[git-user]/[repo].git
~~~

**It must be changed by the `Host alias` from our `config` files like this:**
~~~
url = git@[alias]:[git-user]/[repo].git
~~~
