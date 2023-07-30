import os, sys 

class suppress_stdout_stderr(object):
    
    def __init__(self, suppress=True) -> None:
        self.suppress = suppress
        
    def __enter__(self):
        if not self.suppress: return self

        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_): 
        if not self.suppress: return       
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()

import pickle
from langchain.llms import LlamaCpp
from langchain import LLMChain
from langchain.memory import ConversationBufferWindowMemory
import os
import sys

from langchain.prompts.chat import PromptTemplate
from langchain import FewShotPromptTemplate

is_debug_mode = False

pickle_file_name = 'history.llama.clihelper.pickle'

input_request="Show my current directory"

# Concatenate arguments together in a string
if len(sys.argv) > 1:
    input_request = " ".join(sys.argv[1:])

chain_memory=ConversationBufferWindowMemory(k=5, human_prefix="\nHuman", ai_prefix="\nAssistant", memory_key="history")
resuming_conversation = False

if os.path.isfile(pickle_file_name):
    with open(pickle_file_name, 'rb') as handle:
        chain_memory = pickle.load(handle)
    resuming_conversation = True

n_gpu_layers = 35  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
with suppress_stdout_stderr(suppress=not is_debug_mode):
    model = LlamaCpp(
        model_path="models/7B/ggml-model-q4_0.bin",
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


with open(pickle_file_name, 'wb') as handle:
    pickle.dump(chain.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
