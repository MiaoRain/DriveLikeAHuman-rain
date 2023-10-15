import base64
import json
from rich import print
import sqlite3

from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.output_parsers import ResponseSchema #用于定义响应的结构。
from langchain.output_parsers import StructuredOutputParser #用于解析结构化输出。
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate #用于构建提示模板。

from scenario.scenario import Scenario


class OutputParser:
    def __init__(self, sce: Scenario,llm, temperature: float = 0.0) -> None:
        self.sce = sce
        self.temperature = temperature
        self.llm = llm
        # todo: put into a yaml file
        self.response_schemas = [
            ResponseSchema(
                name="action_id", description=f"output the id(int) of the decision. The comparative table is:  {{ 0: 'change_lane_left', 1: 'keep_speed or idle', 2: 'change_lane_right', 3: 'accelerate or faster',4: 'decelerate or slower'}} . For example, if the ego car wants to keep speed, please output 1 as a int."),
            ResponseSchema(
                name="action_name", description=f"output the name(str) of the decision. MUST consist with previous \"action_id\". The comparative table is:  {{ 0: 'change_lane_left', 1: 'keep_speed', 2: 'change_lane_right', 3: 'accelerate',4: 'decelerate'}} . For example, if the action_id is 3, please output 'Accelerate' as a str."),
            ResponseSchema(
                name="explanation", description=f"Explain for the driver why you make such decision in 40 words.")
        ]#包含三个 ResponseSchema 对象，每个对象都描述了输出的不同部分（动作 ID、动作名称、解释）的结构。
        self.output_parser = StructuredOutputParser.from_response_schemas(
            self.response_schemas)#结构化输出解析器，用于将输出解析为结构化的数据。
        self.format_instructions = self.output_parser.get_format_instructions()#获取格式化指令，并将其存储在 format_instructions 实例变量中。

    def agentRun(self, final_results: dict) -> str:
        print('[green]Output parser is running...[/green]')
        prompt_template = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "parse the problem response follow the format instruction.\nformat_instructions:{format_instructions}\n response: {answer}")
            ],
            input_variables=["answer"],
            partial_variables={"format_instructions": self.format_instructions}
        )
        input = prompt_template.format_prompt(
            answer=final_results['answer']+final_results['thoughts'])
        with get_openai_callback() as cb:
            output = self.llm(input.to_messages())

        self.parseredOutput = self.output_parser.parse(output.content)
        self.dataCommit()
        print("Finish output agent:\n", cb)
        return self.parseredOutput

    def dataCommit(self):
        conn = sqlite3.connect(self.sce.database)
        cur = conn.cursor()
        parseredOutput = json.dumps(self.parseredOutput)
        base64Output = base64.b64encode(
            parseredOutput.encode('utf-8')).decode('utf-8')
        cur.execute(
            """UPDATE decisionINFO SET outputParser ='{}' WHERE frame ={};""".format(
                base64Output, self.sce.frame
            )
        )
        conn.commit()
        conn.close()
