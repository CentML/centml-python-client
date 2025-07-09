from setuptools import setup, find_packages

REQUIRES = []
with open('requirements.txt') as f:
    for line in f:
        line, _, _ = line.partition('#')
        line = line.strip()
        if not line or line.startswith('setuptools'):
            continue
        REQUIRES.append(line)

setup(
    name='centml',
    version='0.4.3',
    packages=find_packages(),
    python_requires=">=3.10",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    entry_points={
        "console_scripts": [
            "centml = centml.cli:cli",
            "ccluster = centml.cli:ccluster",
        ],
    },
    install_requires=REQUIRES
)
