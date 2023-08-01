import pickle
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
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

load_dotenv()
is_debug_mode = os.environ.get("DEBUG_MODE", False) == "True"

filename = os.path.splitext(os.path.basename(__file__))[0]
script_dir = os.path.dirname(os.path.realpath(__file__))
history_file_path = os.path.join(script_dir, f'{filename}.pickle')

input_request="Show my current path"

# Concatenate arguments together in a string
if len(sys.argv) > 1:
    input_request = " ".join(sys.argv[1:])

chain_memory=ConversationBufferWindowMemory(k=2)

if os.path.isfile(history_file_path):
    with open(history_file_path, 'rb') as handle:
        chain_memory = pickle.load(handle)

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    verbose=is_debug_mode
)


template = "You are a helpful assistant that outputs example Linux commands.I will describe what I want to do, and you will reply with a Linux command to accomplish that task. I want you to only reply with the Linux Bash command, and nothing else. Do not write explanations. Only output the command. If you don't have a Linux command to respond with, say you don't know, in an echo command"
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

# get a chat completion from the formatted messages
print(chain.run(input_request))

print(chain.memory.chat_memory.messages)

# Unfortunately langchain's conversation buffer window stores _all_ messages in memory even if it only serves a few `k`. Have to manually trim it. 
chain.memory.chat_memory.messages = chain.memory.chat_memory.messages[-4:]

with open(history_file_path, 'wb') as handle:
    pickle.dump(chain.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
