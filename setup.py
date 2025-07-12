from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DmDSLab",
    version="0.0.0",
    author="Dmatryus Detry",
    author_email="dmatryus.sqrt49@yandex.ru",
    description="Data Science Laboratory Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dmatryus/DmDSLab",
    packages=find_packages(include=['dmdslab']),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.8',
    keywords='data-science machine-learning preprocessing',
    project_urls={
        "Documentation": "https://github.com/Dmatryus/DmDSLab/wiki",
        "Source": "https://github.com/Dmatryus/DmDSLab",
    },
)
