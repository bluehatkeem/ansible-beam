import os
import nltk
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain_pinecone import PineconeVectorStore
from colors import bcolors
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from llm_model import llm, parser, template, prompt, embeddings, memory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# from langchain.chains.llm import LLMChain

load_dotenv()

nltk.download("punkt")

class AnsibleResourceGenerator:
    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX_NAME") 

    def get_user_input(self):
        while True:
            resource_type = input(
                f"{bcolors.OKBLUE}What kind of Ansible resource do you want to create (playbook/role)?{bcolors.ENDC} "
            ).strip().lower()
            if resource_type in ["playbook", "role"]:
                break
            print(f"{bcolors.FAIL}Invalid input. Please enter 'playbook' or 'role'.{bcolors.ENDC}")

        description = input(
            f"{bcolors.OKBLUE}What should the {resource_type} do? {bcolors.ENDC}"
        ).strip()
        return resource_type, description

    def load_docs(self):
        print(f"{bcolors.OKGREEN}Loading playbooks into document loader... {bcolors.ENDC}")
        playbook_path = os.path.join(os.getenv("ANSIBLE_HOME"), "playbooks")
        playbook_loader = DirectoryLoader(
            path=playbook_path, glob="**/*.yml", show_progress=False
        )
        playbooks = playbook_loader.load()
        return playbooks

    def split_docs(self, playbooks, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs = text_splitter.split_documents(playbooks)
        return docs

    def create_vector_store(self, docs):
        db = PineconeVectorStore.from_documents(docs, embeddings, index_name=self.index_name)
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8},
        )
        return retriever

    def define_processing_chain(self, retriever):
        chain = (
            {
                "context": (lambda x: x["description"]) | retriever,
                "description": (lambda x: x["description"]),
                "format_instructions": (lambda x: x["format_instructions"]),
                "resource_type": (lambda x: x["resource_type"]),
            }
            | prompt
            | llm
            | parser
        )

        # chain2 = LLMChain(
        #     llm,
        #     prompt,
        #     output_parser=parser,
        #     verbose=True,
        #     memory=memory,
        # )

        return chain

    def generate_resource(self, chain, description, resource_type):

        def get_session_history(session_id: str) -> BaseChatMessageHistory:

            store = {}

            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        print(f"{bcolors.OKCYAN}Generating Ansible resource ... {bcolors.ENDC}")

        new_resource = RunnableWithMessageHistory(
            chain,
            get_session_history=get_session_history,
            input_messages_key="description",
            history_messages_key="chat_history",
        )
        new_resource = chain.invoke(
            {"description": description, "resource_type": resource_type, "format_instructions": parser.get_format_instructions()},
            config={"configurable": {"session_id": "abc123"}},
        )
        return new_resource

    def save_ansible_resource(self, resource_type, file_name, content):
        output_dir = f"./{resource_type}"
        os.makedirs(output_dir, exist_ok=True)

        if resource_type in ["playbook", "playbooks"]:
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w") as file:
                file.write(content)
            print(
                f"{bcolors.OKGREEN}SUCCESS:{bcolors.ENDC}{resource_type.capitalize()} generated and saved to {file_path}"
            )
        elif resource_type in ["role", "roles"]:
            for subfile_name, subfile_content in content.items():
                subfile_dir = os.path.join(output_dir, file_name)
                role_dir_name = subfile_name.split("/")[0]
                role_dir = os.path.join(subfile_dir, role_dir_name)
                os.makedirs(role_dir, exist_ok=True)
                subfile_path = os.path.join(subfile_dir, subfile_name)
                with open(subfile_path, "w") as subfile:
                    subfile.write(subfile_content)
                print(
                     f"{bcolors.OKGREEN}SUCCESS:{bcolors.ENDC}{subfile_name.split('/')[0].capitalize()} generated and saved to {subfile_path}"
                )

    def run(self):
        resource_type, description = self.get_user_input()
        playbooks = self.load_docs()
        docs = self.split_docs(playbooks)
        retriever = self.create_vector_store(docs)
        chain = self.define_processing_chain(retriever)

        while True: 
            new_resource = self.generate_resource(chain, description, resource_type)

            if resource_type in ["role", "roles"]:
                print(new_resource.output.resource_type)
                print(new_resource.output.file_name)
                print(f"tasks/main.yml:\n{new_resource.output.tasks}")
                print(f"handlers/main.yml:\n{new_resource.output.handlers}")
                print(f"vars/main.yml:\n{new_resource.output.vars}")
                print(f"defaults/main.yml:\n{new_resource.output.defaults}")
                print(f"files/config_file:\n{new_resource.output.files}")
                print(f"meta/main.yml:\n{new_resource.output.meta}")
            elif resource_type == "playbook":
                print(new_resource.output.playbook)

            # Ask if the user wants to make changes
            make_changes = input(
                f"{bcolors.WARNING}Make Changes? (yes/no or y/n): {bcolors.ENDC}"
            ).strip().lower()

            if make_changes in ["yes", "y"]:
                description = input(f"{bcolors.OKBLUE}What would you like to do? {bcolors.ENDC}").strip()
            else:
                break

        confirmation = input(
            f"{bcolors.WARNING}Do you want to save the generated Ansible resource? (yes/no): {bcolors.ENDC}"
        )

        if confirmation.lower() in ["yes", "y"]:
            if resource_type in ["role", "roles"]:
                self.save_ansible_resource(
                    new_resource.output.resource_type,
                    new_resource.output.file_name,
                    {
                        "tasks/main.yml": new_resource.output.tasks,
                        "handlers/main.yml": new_resource.output.handlers,
                        "vars/main.yml": new_resource.output.vars,
                        "defaults/main.yml": new_resource.output.defaults,
                        "files/config_file": new_resource.output.files,
                        "meta/main.yml": new_resource.output.meta,
                    },
                )
            elif resource_type in ["playbook", "playbooks"]:
                self.save_ansible_resource(
                    new_resource.output.resource_type,
                    new_resource.output.file_name,
                    new_resource.output.playbook,
                )
        else:
            print(f"{bcolors.FAIL}Operation cancelled by user.{bcolors.ENDC}")

if __name__ == "__main__":
    generator = AnsibleResourceGenerator()
    generator.run()
