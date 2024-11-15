from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    description = f.read()

setup(
    name='availai',
    version='0.2.0',
    description='A library for computer vision and machine learning',
    long_description=description,
    long_description_content_type='text/markdown',
    author='Vasyl Arsenii',
    author_email='varsenyi@gmail.com',
    packages=find_packages(),
    install_requires=[
        'ultralytics>=8.3.28',
        'wandb>=0.18.6',
        'roboflow>=1.1.49'
    ],
)