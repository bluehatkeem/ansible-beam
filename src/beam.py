import os
import time
import sys
import openai
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import DirectoryLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from typing import Union
from langchain_core.prompts import PromptTemplate
import nltk
from dotdot import print_with_ellipsis as pwe
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.output_parsers import YamlOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from template import template
from colors import bcolors
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_community.llms.ollama import Ollama


nltk.download("punkt")


class PlaybookResource(BaseModel):
    """Used to Generate an Ansible playbook only"""

    resource_type: str = Field(
        description="The type of Ansible resource to generate (playbooks/roles)."
    )
    file_name: str = Field(
        description="The name of the ansible playbook being generated i.e remove_vim.yml"
    )
    playbook: str = Field(description="The generated Ansible playbook in JSON format.")


class RoleResource(BaseModel):
    """Used to Generate an Ansible role only"""

    resource_type: str = Field(
        description="The type of Ansible resource to generate (playbooks/roles)."
    )
    file_name: str = Field(
        description="The name of the ansible role resource dir being generated i.g roles/remove_vim"
    )
    task: str = Field(description="The generated Ansible resource in the tasks/ dir.")
    handlers: str = Field(
        description="The generated Ansible resource in the handler/ dir."
    )
    vars: str = Field(description="The generated Ansible resource in the vars/ dir.")
    defaults: str = Field(
        description="The generated Ansible resource in the defaults/ dir."
    )
    file: str = Field(description="The generated Ansible resource in the files/ dir.")
    meta: str = Field(description="The generated Ansible resource in the meta/ dir.")


class Resource(BaseModel):
    output: Union[PlaybookResource, RoleResource]


# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.4,
)

# llm = Ollama(
#     model="llama3",
#     base_url="http://keemgpt.kubeworld.io:11434"
#     )


parser = YamlOutputParser(pydantic_object=Resource)


# template = template.replace("{resource_type}", "playbook")

prompt = ChatPromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

model = ChatOpenAI(model="gpt-4o")

# initializing the embeddings
embeddings = OpenAIEmbeddings()


def get_user_input():
    resource_type = (
        input(
            f"{bcolors.OKBLUE}What kind of Ansible resource do you want to create (playbook/role)?{bcolors.ENDC} "
        )
        .strip()
        .lower()
    )
    if resource_type not in ["playbook", "role"]:
        print(
            f"{bcolors.FAIL}Invalid input. Please enter 'playbook' or 'role'.{bcolors.ENDC}"
        )
        return get_user_input()

    description = input(
        f"{bcolors.OKBLUE}What should the {resource_type} do? {bcolors.ENDC}"
    ).strip()
    return resource_type, description


def load_docs():
    # tell user that we are now loading roles and playbooks into document loader

    # role_path = os.path.join(os.getcwd(), "roles")
    # role_loader = DirectoryLoader(path=dir_path, glob="**/*.yml", show_progress=True)
    # roles = role_loader.load()

    print(f"{bcolors.OKGREEN}Loading playbooks into document loader... {bcolors.ENDC}")

    playbook_path = os.path.join(os.getcwd(), "playbooks")
    playbook_loader = DirectoryLoader(
        path=playbook_path, glob="**/*.yml", show_progress=False
    )
    playbooks = playbook_loader.load()

    # print("Loading ansible config file into document loader...")
    # ansible_cfg_path = os.path.join(os.getcwd(), ".")
    # ansible_cfg_loader = DirectoryLoader(path=dir_path, glob="**/*.cfg", show_progress=True)
    # ansible_cfg = ansible_cfg_loader.load()

    return playbooks


def split_docs(playbooks, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(playbooks)
    return docs


index_name = "ansible-playbooks"

# def save_ansible_resource(resource_type, file_name, content):
#     output_dir = f"./{resource_type}"
#     os.makedirs(output_dir, exist_ok=True)

#     if resource_type == "playbooks":
#         file_path = os.path.join(output_dir, file_name)
#     else:
#         file_path = os.path.join(output_dir, file_name)
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     with open(file_path, 'w') as file:
#         file.write(content)

#     print(f"{bcolors.OKGREEN}SUCCESS:{bcolors.ENDC}{resource_type.capitalize()} generated and saved to {file_path}")


def save_ansible_resource(resource_type, file_name, content):
    output_dir = f"./{resource_type}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    if resource_type == "playbooks":
        with open(file_path, "w") as file:
            file.write(content)
        print(
            f"{bcolors.OKGREEN}SUCCESS:{bcolors.ENDC}{resource_type.capitalize()} generated and saved to {file_path}"
        )
    elif resource_type == "role":
        for subfile_name, subfile_content in content.items():
            subfile_path = os.path.join(output_dir, subfile_name)
            os.makedirs(os.path.dirname(subfile_path), exist_ok=True)
            with open(subfile_path, "w") as subfile:
                subfile.write(subfile_content)
            print(
                f"{bcolors.OKGREEN}SUCCESS:{bcolors.ENDC}{resource_type.capitalize()} generated and saved to {subfile_path}"
            )
    else:
        # handle other resource types
        pass


def main():
    resource_type, description = get_user_input()
    playbooks = load_docs()
    docs = split_docs(playbooks)

    db = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8},
    )
    chain = (
        {
            "context": (lambda x: x["description"]) | retriever,
            "description": (lambda x: x["description"]),
            "resource_type": (lambda x: x["resource_type"]),
        }
        | prompt
        | llm
        | parser
    )

    print(f"{bcolors.OKCYAN}Generating Ansible resource ... {bcolors.ENDC}")

    # covert descrption into a dictionary

    new_resource = chain.invoke(
        {"description": description, "resource_type": resource_type}
    )

    # Check resource type and print accordingly
    if resource_type == "role":
        print(f"task/main.yml:\n{new_resource.output.task}")
        print(f"handlers/main.yml:\n{new_resource.output.handlers}")
        print(f"vars/main.yml:\n{new_resource.output.vars}")
        print(f"defaults/main.yml:\n{new_resource.output.defaults}")
        print(f"files:\n{new_resource.output.file}")
        print(f"meta/main.yml:\n{new_resource.output.meta}")
    elif resource_type == "playbook":
        print(new_resource.output.playbook)

    # Ask for user confirmation before saving
    confirmation = input(
        f"{bcolors.WARNING}Do you want to save the generated Ansible resource? (yes/no): {bcolors.ENDC}"
    )
    if confirmation.lower() == "yes":
        if new_resource.output.resource_type == "role":
            save_ansible_resource(
                new_resource.output.resource_type,
                new_resource.output.file_name,
                {
                    "task/main.yml": new_resource.output.task,
                    "handlers/main.yml": new_resource.output.handlers,
                    "vars/main.yml": new_resource.output.vars,
                    "defaults/main.yml": new_resource.output.defaults,
                    "files": new_resource.output.file,
                    "meta/main.yml": new_resource.output.meta,
                },
            )
        elif new_resource.output.resource_type == "playbook":
            save_ansible_resource(
                new_resource.output.resource_type,
                new_resource.output.file_name,
                new_resource.output.playbook,
            )
    else:
        print(f"{bcolors.FAIL}Operation cancelled by user.{bcolors.ENDC}")


if __name__ == "__main__":
    main()
