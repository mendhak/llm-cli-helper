
A very basic CLI helper. Describe a bash command and it outputs sample line(s).  

![example](example.gif)

## Set up

### Install the dependencies

Set up dependencies

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Create a file called `.env` and place your OpenAI API key in there. 

```
OPENAI_API_KEY="......................................."
```

### Create an alias

Best used as an alias called `?`.  Add it to ~/.bashrc like so:

```bash
    # add alias
    echo alias ?="\"$(pwd)/.venv/bin/python3 $(realpath clihelper.py)\"" >> ~/.bashrc
    # reload bash
    exec bash
```

## Use it

Try a few commands

```bash
$ ? how much disk space 

df -h

$ ? show top processes by CPU usage

top -o %CPU

```

There is a history, so the next question can be a follow up.  Example:

```bash
$ ? find .pickle files in this directory

find . -type f -name "*.pickle"

$ ? delete them

find . -type f -name "*.pickle" -delete
```


Another example, I didn't like the first output so asked for nc instead.

```bash 
$ ? check if port 443 on example.com is open

echo | telnet example.com 443

$ ? using nc

nc -zv example.com 443


```