from langchain import HuggingFaceHub, LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.chains.conversation.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Env variables
load_dotenv()

# Custom agent
from agents.custom_prompt import CustomPromptTemplate, template
from agents.custom_output_parser import CustomOutputParser

def main():
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1})
    tools = []
    memory = ConversationBufferMemory(memory_key="chat_history")
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = CustomOutputParser()
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools],
        memory=memory,
        verbose=False
    )
    agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)

    while True:
        user_input = input("Human: ")

        if user_input.lower() == "exit":
            print("Closing chat...")
            break

        response = agent.run(user_input)  # Sends user input to chatbot
        print("Chatbot: " + response)

if __name__ == "__main__":
    main()
