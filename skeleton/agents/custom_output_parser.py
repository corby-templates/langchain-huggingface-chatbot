from langchain.agents import AgentOutputParser
from typing import Union
from langchain.schema import AgentAction, AgentFinish
import re

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )