import pickle
from dotenv import load_dotenv
from langchain.llms import LlamaCpp
from langchain import LLMChain
from langchain.memory import ConversationBufferWindowMemory
import os
import sys
from suppress_llamacpp_stderr import suppress_stdout_stderr

from langchain.prompts.chat import PromptTemplate
from langchain import FewShotPromptTemplate

load_dotenv()
is_debug_mode = os.environ.get("DEBUG_MODE", False) == "True"
llama_model_path = os.environ.get("LLAMA_MODEL_PATH", "models/7B/ggml-model-q4_0.bin")

script_dir = os.path.dirname(os.path.realpath(__file__))
history_file_path = os.path.join(script_dir, 'history.llama.clihelper.pickle')


input_request="Show my current directory"

# Concatenate arguments together in a string
if len(sys.argv) > 1:
    input_request = " ".join(sys.argv[1:])

chain_memory=ConversationBufferWindowMemory(k=5, human_prefix="\nHuman", ai_prefix="\nAssistant", memory_key="history")

if os.path.isfile(history_file_path):
    with open(history_file_path, 'rb') as handle:
        chain_memory = pickle.load(handle)

n_gpu_layers = 35  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
with suppress_stdout_stderr(suppress=not is_debug_mode):
    model = LlamaCpp(
        model_path=llama_model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=is_debug_mode,
        stop=["Human:", "\n\n"], 
    )

# create our examples
examples = [
    {
        "query": "List files in the current directory",
        "answer": "\n```\nls\n```"
    }, 
    {
        "query": "Push my git branch up!",
        "answer": "\n```\ngit push origin <branchname>\n```"
    },
    {
        "query": "What is your name?",
        "answer": "\n```\necho Sorry, I don't know a bash command for that.\n```"
    },
    {
        "query": "What OS am I using?",
        "answer": "\n```\ncat /etc/issue\n```"
    }
]

# create a example template
example_template = """Human: {query}

Assistant: 
{answer}

"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix, the prefix is our instructions
prefix = """System: You are a helpful assistant that outputs example Linux commands. I will describe what I want to do, and you will reply with a Linux command inside a unique code block on a new line to accomplish that task. 
I want you to only reply with the Linux Bash command inside a unique code block, and nothing else. 
Do not write explanations or comments. Only output the command inside a unique code block. 
If you don't have a Linux command to respond with, say you don't know, in an echo command. Here are some examples: 
"""
# and the suffix our user input and output indicator
suffix = """
{history}

Human: {query}

Assistant:
"""

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["history", "query"],
    example_separator="\n"
)

chain = LLMChain(llm=model, prompt=few_shot_prompt_template, memory=chain_memory, verbose=is_debug_mode)

response = chain.run(input_request)
print(response)


with open(history_file_path, 'wb') as handle:
    pickle.dump(chain.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
