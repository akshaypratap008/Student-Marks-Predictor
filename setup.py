from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'       #to prevent it from being stored in requirements variable from requirements.txt file


def get_requirements(file_path:str)->List[str]:     
    # function returns all the libraries from requirements.txt file in a list
    '''
    This function will return the list of requirements 
    '''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name = 'mlproject',
    version = '0.0.1',
    author = 'Akshay',
    author_email = 'ap.akshay008@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)