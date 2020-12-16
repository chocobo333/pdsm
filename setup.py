
from setuptools import setup, find_packages


setup(
    name = "pdsm",
    version = "0.1.0",
    description = "",
    author = "chocobo333",
    author_email = "chocobo1355@gmail.com",
    url = "",
    packages = find_packages(),
    entry_points = """
    [console_scripts]
    thesis_test = pdsm._thesis:run
    """,
    install_requires = open("requirements.txt").read().splitlines(),
)