from setuptools import setup, find_packages

setup(
    name='sub_agents',
    version='0.1.0',  # Choose an appropriate version number
    packages=find_packages(),
    install_requires=[
        'google-cloud-vision',
        'google-cloud-storage',
        'google-adk'
    ],
    description='A collection of sub-agents for logistics customer support.',
    author='Your Name',  # Replace with your name or organization
    author_email='your.email@example.com',  # Replace with your email
    # Add more metadata as needed (e.g., license, URL, classifiers)
)
