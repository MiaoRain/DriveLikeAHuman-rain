"""
driver agent 可以通过调用工具来认知驾驶环境，进而做出驾驶决策。
Driver agents can perceive the driving environment by calling tools and make driving decisions accordingly.
"""

from rich import print #终端中美化和格式化文本输出。
import sqlite3 #与SQLite数据库进行交互。

from typing import Union
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType #用于初始化驾驶代理。
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory #这些是用于存储对话和记忆的类。
from langchain.callbacks import get_openai_callback #回调函数，以便将生成的自然语言文本与其他信息一起存储在内存中。
from langchain.agents.tools import Tool

from LLMDriver.callbackHandler import CustomHandler #自定义的回调处理程序。
from scenario.scenario import Scenario
from LLMDriver.agent_propmts import SYSTEM_MESSAGE_PREFIX, SYSTEM_MESSAGE_SUFFIX, FORMAT_INSTRUCTIONS, HUMAN_MESSAGE, TRAFFIC_RULES, DECISION_CAUTIONS
#一些常量和文本，用于定义驾驶代理的提示和规则。

class DriverAgent:
    def __init__(
        self, llm: Union[ChatOpenAI, AzureChatOpenAI, OpenAI], toolModels: list, sce: Scenario,
        verbose: bool = False
    ) -> None:
        self.sce = sce
        self.ch = CustomHandler()#回调处理程序CustomHandler的实例，并将其保存在self.ch属性中。
        self.llm = llm

        self.tools = []
        for ins in toolModels:
            func = getattr(ins, 'inference')
            self.tools.append(
                Tool(name=func.name, description=func.description, func=func)
            )

        # self.memory = ConversationBufferMemory(
        #     memory_key="chat_history", output_key='output'
        # )
        self.memory = ConversationTokenBufferMemory(
            memory_key="chat_history", llm=self.llm, max_token_limit=2048)#用于存储对话和记忆。

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            memory=self.memory,
            agent_kwargs={
                'system_message_prefix': SYSTEM_MESSAGE_PREFIX,
                'syetem_message_suffix': SYSTEM_MESSAGE_SUFFIX,
                'human_message': HUMAN_MESSAGE,
                'format_instructions': FORMAT_INSTRUCTIONS,
            },
            handle_parsing_errors="Check your output and make sure it conforms the format instructions!",
            max_iterations=12,
            early_stopping_method="generate",
        )

    def agentRun(self, last_step_decision: dict):
        print(f'Decision at frame {self.sce.frame} is running ...')
        print('[green]Driver agent is running ...[/green]')
        self.ch.memory = []
        if last_step_decision is not None and "action_name" in last_step_decision:
            last_step_action = last_step_decision["action_name"]
            last_step_explanation = last_step_decision["explanation"]
        else:
            last_step_action = "Not available"
            last_step_explanation = "Not available"
        with get_openai_callback() as cb:#回调函数，将agent生成的自然语言文本与其他信息一起存储在内存中，以备后续使用。
            #self.agent.run 是一个方法调用，用于执行驾驶代理的决策过程。具体来说，这个方法通过与预定义的规则和逻辑来生成驾驶决策，该决策是以自然语言文本的形式输出的。
            #self.agent.run 方法接受一个包含文本提示和其他信息的字符串作为输入。这个字符串提供了驾驶代理在当前决策步骤中需要考虑的上下文、规则和任务。
            #总之，self.agent.run 方法是驾驶代理执行决策生成过程的入口点，它接受输入提示并使用内部的逻辑和回调函数来生成驾驶决策的文本描述。这个文本描述可以作为代理的最终输出，用于驾驶决策的记录和分析。
            self.agent.run(
                f"""
                You, the 'ego' car, are now driving a car on a highway. You have already drive for {self.sce.frame} seconds.
                The decision you made LAST time step was `{last_step_action}`. Your explanation was `{last_step_explanation}`. 
                Here is the current scenario: \n ```json\n{self.sce.export2json()}\n```\n. 
                Please make decision for the `ego` car. You have to describe the state of the `ego`, then analyze the possible actions, and finally output your decision. 

                There are several rules you need to follow when you drive on a highway:
                {TRAFFIC_RULES}

                Here are your attentions points:
                {DECISION_CAUTIONS}
                
                Let's think step by step. Once you made a final decision, output it in the following format: \n
                ```
                Final Answer: 
                    "decision":{{"ego car's decision, ONE of the available actions"}},
                    "expalanations":{{"your explaination about your decision, described your suggestions to the driver"}}
                ``` \n
                """,
                callbacks=[self.ch]
            )

        print(cb)
        print('[cyan]Final decision:[/cyan]')
        print(self.ch.memory[-1])
        self.dataCommit()

    def exportThoughts(self):
        output = {}
        output['thoughts'], output['answer'] = self.ch.memory[-1].split(
            'Final Answer:')
        return output

    def dataCommit(self):
        scenario = self.sce.export2json()
        thinkAndThoughts = '\n'.join(self.ch.memory[:-1])
        finalAnswer = self.ch.memory[-1]
        conn = sqlite3.connect(self.sce.database)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO decisionINFO VALUES (?,?,?,?,?)""",
            (self.sce.frame, scenario, thinkAndThoughts, finalAnswer, '')
        )
        conn.commit()
        conn.close()
