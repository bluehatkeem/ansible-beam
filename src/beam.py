
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain_pinecone import PineconeVectorStore
import nltk
from colors import bcolors
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from llm_model import llm
from llm_model import parser
from llm_model import template
from llm_model import prompt
from llm_model import embeddings





nltk.download("punkt")







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
    # file_path = os.path.join(output_dir, file_name)

    if resource_type == "playbook":
        # with open(file_path, "w") as file:
        #     file.write(content)
        print(
            f"{bcolors.OKGREEN}SUCCESS:{bcolors.ENDC}{resource_type.capitalize()} generated and saved to /file_path/"
        )
    elif resource_type == "role":
        for subfile_name, subfile_content in content.items():
            subfile_dir = os.path.join(output_dir, file_name)
            os.makedirs(subfile_dir, exist_ok=True)
            subfile_path = os.path.join(subfile_dir, subfile_name)
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
        print(new_resource.output.resource_type)
        print(new_resource.output.file_name)
        print(f"task/main.yml:\n{new_resource.output.tasks}")
        print(f"handlers/main.yml:\n{new_resource.output.handlers}")
        print(f"vars/main.yml:\n{new_resource.output.vars}")
        print(f"defaults/main.yml:\n{new_resource.output.defaults}")
        print(f"files:\n{new_resource.output.files}")
        print(f"meta/main.yml:\n{new_resource.output.meta}")
    elif resource_type == "playbook":
        print(new_resource.output.playbook)

    # Ask for user confirmation before saving
    confirmation = input(
        f"{bcolors.WARNING}Do you want to save the generated Ansible resource? (yes/no): {bcolors.ENDC}"
    )
    if confirmation.lower() == "yes":
        if resource_type == "role":
            save_ansible_resource(
                new_resource.output.resource_type,
                new_resource.output.file_name,
                {
                    "tasks/main.yml": new_resource.output.tasks,
                    "handlers/main.yml": new_resource.output.handlers,
                    "vars/main.yml": new_resource.output.vars,
                    "defaults/main.yml": new_resource.output.defaults,
                    "files/": new_resource.output.files,
                    "meta/main.yml": new_resource.output.meta,
                },
            )
        elif resource_type == "playbook":
            save_ansible_resource(
                new_resource.output.resource_type,
                new_resource.output.file_name,
                new_resource.output.playbook,
            )
    else:
        print(f"{bcolors.FAIL}Operation cancelled by user.{bcolors.ENDC}")


if __name__ == "__main__":
    main()
