# 使用langchain加载中文模型
# 继承LLM，并重写_llm_type、_call、_identifying_params方法

# ##################langchain模型###################################
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI, ChatVertexAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")
response_schemas = [gift_schema,
                    delivery_days_schema,
                    price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)
os.environ["OPENAI_API_KEY"] = 'sk-mfSpwdDUbFLA1Sq0nGldKy5IY9StuRr9cbcuVRQWcsxsV2Kl'
messages = prompt_template.format_messages(text=customer_review)
memory = ConversationBufferMemory()
chat = ChatOpenAI(temperature = 0,
model_name = 'gpt-3.5-turbo-16k-0613',
openai_api_base = "https://api.chatanywhere.com.cn/v1",)
response = chat(messages)

output_dict = output_parser.parse(response.content)

# ##################langchain模型###################################




# ##################下載hugging_face_hub中的模型###################################
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="openbmb/cpm-bee-1b", allow_patterns=["*.json", "pytorch_model.bin", "*.py", "vocab.txt"], local_dir="./cpm-bee-1b/")
# #################################################################################

################################################################################################################
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# class ModelLoader:
#     def __init__(self, model_name_or_path):
#         self.model_name_or_path = model_name_or_path
#         self.model, self.tokenizer = self.load()
#     def load(self):
#         tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True)#.cuda()
#         return model, tokenizer
# # 指定本地模型的路径
# model_name_or_path = ".\\cpm-bee-1b"
# modelLoader = ModelLoader(model_name_or_path)
# from typing import Any, List, Mapping, Optional
# from langchain.llms.base import LLM
# class CpmBee(LLM):
#     @property
#     def _llm_type(self) -> str:
#         return "cpm-bee-1B"
#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         print(prompt)
#         prompt = json.loads(prompt)
#         tokenizer = modelLoader.tokenizer
#         model = modelLoader.model
#         result = model.generate(prompt, tokenizer)
#         if len(result) == 1:
#             return result[0]["<ans>"]
#         return "对不起，我没有理解你的意思"
#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         params_dict = {
#             "test": "这里是参数字典",
#         }
#         return params_dict
# prompt = {"input": "今天天气真不错", "question": "今天天气怎么样？", "<ans>": ""}
# cpmBee = CpmBee()
# print(cpmBee)
# print(cpmBee(json.dumps(prompt, ensure_ascii=False)))
# """
# CpmBee
# Params: {'test': '这里是参数字典'}
# {"input": "今天天气真不错", "question": "今天天气怎么样？", "<ans>": ""}
# 好
# """
###############################################################################################################
# from typing import Any, List, Mapping, Optional
#
# from langchain.callbacks.manager import CallbackManagerForLLMRun
# from langchain.llms.base import LLM
#
#
# class CustomLLM(LLM):
#     n: int
#
#     @property
#     def _llm_type(self) -> str:
#         return "custom"
#
#     def _call(
#             self,
#             prompt: str,
#             stop: Optional[List[str]] = None,
#             run_manager: Optional[CallbackManagerForLLMRun] = None,
#     ) -> str:
#         if stop is not None:
#             raise ValueError("stop kwargs are not permitted.")
#         return prompt[:self.n]
#
#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         """Get the identifying parameters."""
#         return {"n": self.n}
# llm = CustomLLM(n=10)
# llm("This is a foobar thing")
# print(llm)



# # 导入所需的库
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # 从预训练模型加载 tokenizer
# tokenizer = AutoTokenizer.from_pretrained("openbmb/cpm-bee-1b", trust_remote_code=True)
#
# # 从预训练模型加载模型，将模型放在 CUDA 设备上以加速推理
# model = AutoModelForCausalLM.from_pretrained("openbmb/cpm-bee-1b", trust_remote_code=True).cuda()
#
# # 生成文本示例1
# # 输入："今天天气真"
# # "<ans>" 是占位符，留空
# result = model.generate({"input": "今天天气真", "<ans>": ""}, tokenizer)
#
# # 打印生成的文本
# print(result)

# 生成文本示例2
# 输入："今天天气真不错"
# "question" 是额外的上下文信息
# "<ans>" 是占位符，留空
# result = model.generate({"input": "今天天气真不错", "question": "今天天气怎么样？", "<ans>": ""}, tokenizer)
#
# # 打印生成的文本
# print(result)



# import time
# import litellm
# from litellm import completion
# from litellm.caching import Cache
# litellm.cache = Cache(type="hosted")
#
# start_time = time.time()
# embedding1 = embedding(model="text-embedding-ada-002", input=["hello from litellm"*5], caching=True)
# end_time = time.time()
# print(f"Embedding 1 response time: {end_time - start_time} seconds")
#
# start_time = time.time()
# embedding2 = embedding(model="text-embedding-ada-002", input=["hello from litellm"*5], caching=True)
# end_time = time.time()
# print(f"Embedding 2 response time: {end_time - start_time} seconds")


# import os
# import wget
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
#
# # Specify the model name and output directory
# model_name = "Baichuan2-7B-Chat"
# output_dir = "E:/baichuan-13b"
#
# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)
#
# # Download the model configuration file
# config = AutoConfig.from_pretrained(model_name)
# config.save_pretrained(output_dir)
#
# # Download the model weights
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model.save_pretrained(output_dir)
#
# # Download the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(output_dir)
#
# # Optionally, you can download additional files associated with the model if available
#
# # Inform the user that the download is complete
# print(f"Model downloaded to {output_dir}")












# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", device_map="auto", trust_remote_code=True)
# inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
# inputs = inputs.to('cuda:0')
# pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
# print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))


# from langchain.tools import BaseTool
# from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
# import yaml,os
# from langchain.agents import initialize_agent, AgentType #用于初始化驾驶代理。
# # 搜索工具
# class SearchTool(BaseTool):
#     name = "Search"
#     description = "如果我想知道天气，'鸡你太美'这两个问题时，请使用它"
#     return_direct = True  # 直接返回结果
#
#     def _run(self, query: str) -> str:
#         print("\nSearchTool query: " + query)
#         return "这个是一个通用的返回"
#
#     async def _arun(self, query: str) -> str:
#         raise NotImplementedError("暂时不支持异步")
#
# # 计算工具
# class CalculatorTool(BaseTool):
#     name = "Calculator"
#     description = "如果是关于数学计算的问题，请使用它"
#
#     def _run(self, query: str) -> str:
#         print("\nCalculatorTool query: " + query)
#         return "3"
#
#     async def _arun(self, query: str) -> str:
#         raise NotImplementedError("暂时不支持异步")
#
# OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
# if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
#     os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
#     os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['AZURE_API_VERSION']
#     os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['AZURE_API_BASE']
#     os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['AZURE_API_KEY']
#     llm = AzureChatOpenAI(
#         deployment_name=OPENAI_CONFIG['AZURE_MODEL'],
#         temperature=0,
#         max_tokens=1024,
#         request_timeout=60
#     )
# elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
#     os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
#
# llm = ChatOpenAI(temperature=0)
# tools = [SearchTool(), CalculatorTool()]
# agent = initialize_agent(
#     tools, llm, agent="zero-shot-react-description", verbose=True)
#
# print("答案：" + agent.run("告诉我10的3次方是多少?"))
# print("答案：" + agent.run("告诉我'hello world'是什么意思"))

# PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
# FORMAT_INSTRUCTIONS = """Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question"""
# SUFFIX = """Begin!
#
# Question: {input}
# Thought:{agent_scratchpad}"""