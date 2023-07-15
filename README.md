
A very basic CLI helper. Describe a bash command and it outputs sample line(s).  

## Set up

Set up dependencies

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Place your OpenAI API key in a file called `.env` 

## Usage

Best use is as an alias called `?`.  Add it to ~/.bashrc like so:

```bash
    echo alias ?="\"$(pwd)/.venv/bin/python3 $(realpath clihelper.py)\"" >> ~/.bashrc
    # Equivalent of alias ?=/home/mendhak/Projects/gpt-cli-helper/.venv/python3 /home/mendhak/Projects/gpt-cli-helper/clihelper.py
```

Reload bash and test it

```bash
$ ? find .pickle files in this directory
find . -type f -name "*.pickle"
$ ? delete them
find . -type f -name "*.pickle" -delete
```