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
    version='0.1.0',
    packages=find_packages(),
    scripts=['bin/centml', 'bin/ccluster'],
    install_requires=REQUIRES
)

