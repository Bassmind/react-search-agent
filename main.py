from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain_core.output_parsers import PydanticOutputParser  
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

# Defining the Tools that the Agent will use.
tools = [TavilySearch()]
# Defining the LLM
llm = ChatOpenAI(model="gpt-4")
# Calling the Hub to get a ReAct prompt in the form of a PromptTemplate.
#react_prompt = hub.pull("hwchase17/react")

#OutputParser from Pydantic, using our own Schema from schemas.py
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

#PromptTemplate object usando el prompt de prompt.py
#Las format_instructions son usadas para decirle al LLM que regrese todo con formato JSON que se usara para transformar en Pydantic Object
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input","agent_scratchpad","tool_names"]).partial(format_instructions=output_parser.get_format_instructions())

# Agent created with the above elements (only creation)
#agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)


# Agent execution/Agent Runtime.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#RunnableLambda to extract "output" from x
extract_output = RunnableLambda(lambda x: x["output"])
#RunnableLambda to parse x to Pydantic
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

#Calling the AgentExecutor, extracting the output and parse it to Pydantic
chain = agent_executor | extract_output | parse_output


def main():
    print("Hello from react-search-agent!")

    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for a Remote Java Developer using Java 8 in Mexico on Linkedin and list their details"
        }
    )

    print(result)


if __name__ == "__main__":
    main()
