
A basic CLI lookup tool. Describe a bash command and it outputs sample line(s) by quering LLMs. It can make use of OpenAI (GPT 3.5) or Llama.cpp models.  

![example](example.gif)

## Usage

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

## Set up

Set up dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make a copy of the `.env.sample` file and call it `.env`

## OpenAI setup (simple)

Get an API key from your [OpenAI account](https://platform.openai.com/account/api-keys). Place it in the `.env` file

```
OPENAI_API_KEY="......................................."
```

There is a small cost associated with OpenAI calls, so it's a good idea to [set monthly limits](https://platform.openai.com/account/billing/limits) on usage.


### Create an alias

The application is best used as an alias called `?`.  Add it to ~/.bashrc like so:

```bash
# add alias
echo alias ?="\"$(pwd)/.venv/bin/python3 $(realpath openai.clihelper.py)\"" >> ~/.bashrc
# reload bash
exec bash
```

Now start using `?`

## Llama.cpp models (uses GPU)

Llama.cpp is a fast way of running local LLMs on your own computer. It is very fast with GPUs which will be my focus. It is free to use.

### Install dependencies

First, ensure that [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) is installed. 
After installing Cuda, add it to your path and reload bash:  

```bash
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
exec bash
# test that it worked: 
nvcc --version
```    

Next install the cmake and Python dependencies, and build one specific package with GPU support. 

```bash
sudo apt install make cmake
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force --no-cache
```

### Get some models

Because Llama is open, there are many Llama models you can choose from. Llama.cpp requires models to be in the GGML format. Here are some I tested with: 

[TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) - `llama-2-7b-chat.ggmlv3.q4_0.bin`   
[TheBloke/StableBeluga-7B-GGML](https://huggingface.co/TheBloke/StableBeluga-7B-GGML) -  `stablebeluga-7b.ggmlv3.q4_0.bin`  

 Download these then set the path to the model in the .env file. Example: 

     LLAMA_MODEL_PATH="./models/7B/stablebeluga-7b.ggmlv3.q4_0.bin"

### Create an alias

The application is best used as an alias called `?`.  Add it to ~/.bashrc like so:

```bash
# add alias
echo alias ?="\"$(pwd)/.venv/bin/python3 $(realpath llamacpp.clihelper.py)\"" >> ~/.bashrc
# reload bash
exec bash
```

Now start using `?`

## Implementation notes

This was made using [langchain](https://python.langchain.com/docs/get_started/introduction.html), a library that helps make calls to large language models (LLMs) and process its output. 

In this case I did a 'few shot', which is a way of showing the LLM a few examples of questions and the kind of answers to generate. 

I chose the `gpt-3.5-turbo` model which is the cheapest on OpenAI currently.  

