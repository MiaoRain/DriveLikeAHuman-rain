# # ##################langchain模型###################################
# from typing import Any, List, Mapping, Optional
# from langchain.callbacks.manager import CallbackManagerForLLMRun
# from langchain.llms.base import LLM
# import os
# import yaml
# import numpy as np
# import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
# from langchain.chat_models import AzureChatOpenAI, ChatOpenAI, ChatVertexAI
# from langchain.output_parsers import ResponseSchema
# from langchain.output_parsers import StructuredOutputParser
#
# import os
#
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file
#
# import warnings
# warnings.filterwarnings('ignore')
# #%%
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
#
# os.environ["OPENAI_API_KEY"] = 'sk-mfSpwdDUbFLA1Sq0nGldKy5IY9StuRr9cbcuVRQWcsxsV2Kl'
# llm = ChatOpenAI(temperature = 0,
# # model_name = 'gpt-3.5-turbo-16k-0613',
# openai_api_base = "https://api.chatanywhere.com.cn/v1",)
#
# # llm = ChatOpenAI(temperature=0.0)
# memory = ConversationBufferMemory()
# conversation = ConversationChain(
#     llm=llm,
#     memory = memory,
#     verbose=True
# )
# #%%
# conversation.predict(input="Hi, my name is Andrew")
# #%%
# conversation.predict(input="What is 1+1?")
# #%%
# conversation.predict(input="What is my name?")
# #%%
# #print(memory.buffer)
# #%%
# memory.load_memory_variables({})
# #%%
# memory = ConversationBufferMemory()
# #%%
# memory.save_context({"input": "Hi"},
#                     {"output": "What's up"})
# #%%
# # print(memory.buffer)
# #%%
# memory.load_memory_variables({})
# #%%
# memory.save_context({"input": "Not much, just hanging"},
#                     {"output": "Cool"})
# #%%
# memory.load_memory_variables({})





import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")
os.environ["OPENAI_API_KEY"] = 'sk-mfSpwdDUbFLA1Sq0nGldKy5IY9StuRr9cbcuVRQWcsxsV2Kl'
#%% md
## Built-in LangChain tools
#%%
#!pip install -U wikipedia
#%%
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
import langchain#%%
llm = ChatOpenAI(temperature = 0,
openai_api_base = "https://api.chatanywhere.com.cn/v1",)
# model_name = 'gpt-3.5-turbo-16k-0613',
#%%
tools = load_tools(["llm-math","wikipedia"], llm=llm)
#%%
agent= initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
#%%
langchain.debug = True
agent("What is the 25% of 300?")
#%%
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question)
#%% md
# ## Python Agent
# #%%
# agent = create_python_agent(
#     llm,
#     tool=PythonREPLTool(),
#     verbose=True
# )
# #%%
# customer_list = [["Harrison", "Chase"],
#                  ["Lang", "Chain"],
#                  ["Dolly", "Too"],
#                  ["Elle", "Elem"],
#                  ["Geoff","Fusion"],
#                  ["Trance","Former"],
#                  ["Jen","Ayai"]
#                 ]
# #%%
# agent.run(f"""Sort these customers by \
# last name and then first name \
# and print the output: {customer_list}""")
# #%% md
# #### View detailed outputs of the chains
# #%%
# import langchain
# langchain.debug=True
# agent.run(f"""Sort these customers by \
# last name and then first name \
# and print the output: {customer_list}""")
# langchain.debug=False
# #%% md
# ## Define your own tool
# #%%
# #!pip install DateTime
# #%%
# from langchain.agents import tool
# from datetime import date
# #%%
# @tool
# def time(text: str) -> str:
#     """Returns todays date, use this for any \
#     questions related to knowing todays date. \
#     The input should always be an empty string, \
#     and this function will always return todays \
#     date - any date mathmatics should occur \
#     outside this function."""
#     return str(date.today())
# #%%
# agent= initialize_agent(
#     tools + [time],
#     llm,
#     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     handle_parsing_errors=True,
#     verbose = True)
# #%% md
# # **Note**:
# #
# # The agent will sometimes come to the wrong conclusion (agents are a work in progress!).
# #
# # If it does, please try running it again.
# #%%
# try:
#     result = agent("whats the date today?")
# except:
#     print("exception on external access")
# #%%
#
#





