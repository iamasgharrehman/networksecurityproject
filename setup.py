'''
The setup.py file is an essential part of package and distributing
python projects it is used by setuptools (
or distutils in older python versions) to define the configuration
of project, such as metadata, dependencies and more
'''

from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    this function will return list of requiremnets
    
    """
    requirements_lst:List[str] = []
    try:
        with open('requirements.txt') as file:
            # readlines from the file
            lines=file.readlines()

            ## process each line
            for line in lines:
                requirement= line.strip()
                 ## ignore empty lines and -e.

                if requirement and requirement != '-e .':
                    requirements_lst.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirements_lst


setup(
    name='NetworkSecurityProject',
    version='0.0.1',
    author='Krish Naik',
    author_email='asghareme94@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements()
)
