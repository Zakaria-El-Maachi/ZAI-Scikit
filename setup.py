from setuptools import setup, find_packages

setup(
    name='ZAIScikit',
    version='0.1.0',
    author='Zakaria',
    author_email='zakaria.elmaachi19@gmail.com',
    description='A Machine learning library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Zakaria-El-Maachi/ZAI-Scikit',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
