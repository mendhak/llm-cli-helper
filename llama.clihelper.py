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

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

is_debug_mode = False


input_request="Show my current path"

# Concatenate arguments together in a string
if len(sys.argv) > 1:
    input_request = " ".join(sys.argv[1:])

chain_memory=ConversationBufferWindowMemory(k=2)
resuming_conversation = False

if os.path.isfile('history.clihelper.pickle'):
    with open('history.clihelper.pickle', 'rb') as handle:
        chain_memory = pickle.load(handle)
    resuming_conversation = True

n_gpu_layers = 35  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
with suppress_stdout_stderr(suppress=True):
    model = LlamaCpp(
        model_path="models/7B/ggml-model-q4_0.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        verbose=is_debug_mode,
    )


template = "You are a helpful assistant that outputs example Linux commands. I will describe what I want to do, and you will reply with a Linux command to accomplish that task. I want you to only reply with the Linux Bash command, and nothing else. Do not write explanations. Only output the command. If you don't have a Linux command to respond with, say you don't know, in an echo command"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human_1 = HumanMessagePromptTemplate.from_template("List files in the current directory")
example_ai_1 = AIMessagePromptTemplate.from_template("\nls\n")
example_human_2 = HumanMessagePromptTemplate.from_template("Push my git branch up")
example_ai_2 = AIMessagePromptTemplate.from_template("\ngit push origin <branchname>\n")
example_human_3 = HumanMessagePromptTemplate.from_template("What is your name?")
example_ai_3 = AIMessagePromptTemplate.from_template("\necho Sorry, I don't know a bash command for that.\n")
human_template = "{history}\n{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_human_1, example_ai_1, example_human_2, example_ai_2, example_human_3, example_ai_3, human_message_prompt]
)



chain = LLMChain(llm=model, prompt=chat_prompt, memory=chain_memory, verbose=is_debug_mode)

print(chain.run(input_request))


with open('history.clihelper.pickle', 'wb') as handle:
    pickle.dump(chain.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
