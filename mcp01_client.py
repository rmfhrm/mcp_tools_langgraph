from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import asyncio

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "test": {
                "command": "python",
                "args": ["./mcp02_server.py"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    # agent = create_react_agent("openai:gpt-4o", tools)
    agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools)
    # response = await agent.ainvoke({"messages": "what's 3 + 5?"})
    response = await agent.ainvoke({"messages": "연봉 5천만원 거주자의 소득세는 얼마인가요?"})
    final_message = response['messages'][-1]
    print(final_message)

if __name__ == '__main__':
    asyncio.run(main())