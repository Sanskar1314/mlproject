# Useful in building or application as a package which can bes used by anyone

from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Sanskar',
    author_email='sanskargupta1314@gmail.com',
    packages=find_packages(), # it finds the folder with file '__init__.py'
    install_requires=get_requirements('requirements.txt') # it installs the required packages
)