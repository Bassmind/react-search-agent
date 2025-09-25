from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

#Defining the Tools that the Agent will use.
tools = [TavilySearch()]
#Defining the LLM
llm = ChatOpenAI(model="gpt-4")
#Calling the Hub to get a ReAct prompt in the form of a PromptTemplate.
react_prompt = hub.pull("hwchase17/react")

#Agent created with the above elements (only creation)
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

#Agent execution/Agent Runtime.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chain = agent_executor

def main():
    print("Hello from react-search-agent!")

    result = chain.invoke(input = {
        "input": "Search for 3 job postings for a Remote Java Developer using Java 8 in Mexico on Linkedin and list their details"
    })

    print(result)


if __name__ == "__main__":
    main()
