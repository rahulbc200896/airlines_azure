from setuptools import setup,find_packages
from typing import List

Hyphen_E_DOT= '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path,'r') as file:
        lines = file.readlines()
        requirements = [req.replace("\n","") for req in lines]
        if Hyphen_E_DOT in requirements:
            requirements.remove(Hyphen_E_DOT)

    return requirements

setup(
    name='Airlines',
    version='0.0.1',
    author='rahul b c',
    author_email='rahulbc17@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)