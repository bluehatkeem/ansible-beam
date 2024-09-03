from setuptools import setup, find_packages

setup(
    name='ansible-beam',
    version='1.0.0',
    description='A project to generate Ansible playbooks and roles using AI',
    author='BlueHat Keem',
    author_email='keem@kubeworld.io',
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'pydantic',
        'nltk',
        'langchain',
        'langchain-community',
        'openai'
    ],
)
