from pathlib import Path

from setuptools import (
    find_namespace_packages,  # NOQA: F401 (imported but unused)
)
from setuptools import setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent

with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = ["mkdocs", "mkdocstrings", "mkdocstrings-python"]

style_packages = ["black", "flake8", "isort"]

setup(
    name="sfai",
    version=0.1,
    description="MLOps with Regression Task",
    author="Amaury Maillard",
    author_email="amaury.maill@gmail.com",
    url="",
    python_requires=">=3.7",
    install_requires=[required_packages],
    extras_require={"devs": docs_packages + style_packages, "docs": docs_packages},
)
