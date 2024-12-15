from setuptools import setup, find_packages

with open('requirements.txt') as file:
    requirements = file.readlines()

setup(
    name = 'nba_draft',
    version = '0.0.1',
    packages = find_packages(),
    install_requires = requirements,
    author = 'Morgan Kurth and Josh Bergstrom',
    author_email='mkurth@byu.edu',
    long_description= 'This package provides tools for gathering and analyzing NBA draft data, including college-related statistics. It pulls data on NBA draft picks, enhances the dataset with additional college information, and offers various analysis methods, such as statistical summaries, data wrangling, and machine learning models (e.g., KNN, Decision Trees). The package helps users explore player performance, predict draft outcomes, and gain insights into the influence of college performance on NBA success.'
)