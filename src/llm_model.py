import os
import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.output_parsers import YamlOutputParser
from langchain_community.llms.ollama import Ollama
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Union
from template import template, contextualize_q_system_prompt
from langsmith import traceable
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

load_dotenv()

# Define the PlaybookResource class
class PlaybookResource(BaseModel):
    """Used to Generate an Ansible playbook only"""

    resource_type: str = Field(
        description="The type of resource. should only be 'playbooks'."
    )
    file_name: str = Field(
        description="The name of the ansible playbook being generated."
    )
    playbook: str = Field(description="The generated Ansible playbook in JSON format.")


# Define the RoleResource class
class RoleResource(BaseModel):
    """Used to Generate an Ansible role only"""

    resource_type: str = Field(
        description="The type of Ansible resource to generate. should always be 'roles'."
    )
    file_name: str = Field(
        description="The name of the ansible role resource dir being generated."
    )
    tasks: str = Field(description="The generated Ansible resource in the tasks/ dir.")
    handlers: str = Field(
        description="The generated Ansible resource in the handler/ dir."
    )
    vars: str = Field(description="The generated Ansible resource in the vars/ dir.")
    defaults: str = Field(
        description="The generated Ansible resource in the defaults/ dir."
    )
    files: str = Field(description="The generated Ansible resource in the files/ dir.")
    meta: str = Field(description="The generated Ansible resource in the meta/ dir.")


# Define the Resource class
class Resource(BaseModel):
    output: Union[PlaybookResource, RoleResource]


# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get the model type from environment variables
model_type = os.getenv("CHAT_MODEL_TYPE", "gpt-4o")

# Initialize the appropriate chat model based on the environment variable
if model_type.startswith("gpt"):
    llm = ChatOpenAI(
        model=model_type,
        temperature=0,
    )
elif model_type.startswith("claude"):
    llm = ChatAnthropic(
        model=model_type,
    )
else:
    raise ValueError(f"Unsupported model type: {model_type}")

# Initialize YamlOutputParser instance
parser = YamlOutputParser(pydantic_object=Resource)

# Prompt for the system to contextualize the question

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Initialize ChatPromptTemplate instance
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template),
    ("placeholder", "{chat_history}"),
    ("human", "{description}"), 
])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,

)



# Initialize Pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

# Initialize OpenAIEmbeddings instance
embeddings = OpenAIEmbeddings()
