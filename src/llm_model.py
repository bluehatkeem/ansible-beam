import os
import openai
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.output_parsers import YamlOutputParser
from langchain_community.llms.ollama import Ollama
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Union
from template import template
from langsmith import traceable

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

# Initialize ChatOpenAI instance
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
)

# Initialize YamlOutputParser instance
parser = YamlOutputParser(pydantic_object=Resource)

# Initialize ChatPromptTemplate instance
prompt = ChatPromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Initialize Pinecone API key
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone instance
pc = Pinecone(api_key=pinecone_api_key)

# Initialize ChatOpenAI instance
model = ChatOpenAI(model="gpt-4o")

# Initialize OpenAIEmbeddings instance
embeddings = OpenAIEmbeddings()
