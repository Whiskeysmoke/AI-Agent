import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain.agents import create_agent

load_dotenv()

def get_date():
    """Get the current date"""
    return datetime.now().strftime("%Y-%m-%d")

llm = ChatOllama(model="qwen2.5:3b")

system_prompt = """
You are a helpful assistant.
Use the get_date tool if the user is asking about today's date.
"""

agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)
user_query = input("Enter a query: ")
response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

print(response['messages'][-1].content)