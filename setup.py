from setuptools import setup, find_packages

setup(
    name='centml',
    version='0.1.0',
    packages=find_packages(),
    python_requires="<3.12",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    entry_points={
        "console_scripts": [
            "centml = centml.cli:cli",
            "ccluster = centml.cli:ccluster",
        ],
    },
    install_requires=[
        "fastapi>=0.103.0",
        "uvicorn>=0.23.0",
        "python-multipart>=0.0.6",
        "pydantic-settings==2.0.*",
        "Requests==2.32.2",
        "tabulate>=0.9.0",
        "pyjwt>=2.8.0",
        "cryptography==43.0.1",
        "prometheus-client>=0.20.0",
        "scipy>=1.6.0",
        "scikit-learn>=1.5.1",
        "platform-api-python-client==0.1.0",
    ],
)
