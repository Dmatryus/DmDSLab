from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Основные зависимости
install_requires = [
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.0.0",
]

# Дополнительные зависимости
extras_require = {
    "uci": ["ucimlrepo>=0.0.7"],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
    ],
    "notebook": [
        "jupyter>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
}

# Все дополнительные зависимости
extras_require["all"] = list(set().union(*extras_require.values()))

setup(
    name="DmDSLab",
    version="1.0.0",
    author="Dmatryus Detry",
    author_email="dmatryus.sqrt49@yandex.ru",
    description="Data Science Laboratory Toolkit - инструменты для эффективных исследований",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dmatryus/DmDSLab",
    project_urls={
        "Documentation": "https://github.com/Dmatryus/DmDSLab/wiki",
        "Source": "https://github.com/Dmatryus/DmDSLab",
        "Bug Tracker": "https://github.com/Dmatryus/DmDSLab/issues",
        "Changelog": "https://github.com/Dmatryus/DmDSLab/blob/main/CHANGELOG.md",
    },
    packages=find_packages(include=["dmdslab", "dmdslab.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "data-science",
        "machine-learning",
        "datasets",
        "uci",
        "data-preprocessing",
        "research-tools",
    ],
    entry_points={
        "console_scripts": [
            "dmdslab-init-uci=dmdslab.scripts.initialize_uci_database:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dmdslab": ["datasets/db/*.db"],
    },
)
