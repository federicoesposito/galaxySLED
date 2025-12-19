# galaxySLED: a code to reproduce and fit a galaxy CO SLED
# Copyright (C) 2024  Federico Esposito
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Setup
setup(
    name="galaxysled",
    version="0.1.9",
    description="A code to reproduce and fit a galaxy CO SLED",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/federicoesposito/galaxySLED",
    author="Federico Esposito",
    author_email="federico.esposito7@unibo.it",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Astronomers",
        "Topic :: Astronomical Software :: Build Tools",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "": ["resources/*.csv"],
        "": ["docs/*.ipynb"],
    },
    python_requires=">=3.6, <4",
    install_requires=[
        "anyio>=3.6.2",
        "argon2-cffi>=21.3.0",
        "argon2-cffi-bindings>=21.2.0",
        "async-generator>=1.10",
        "attrs>=22.2.0",
        "Babel>=2.11.0",
        "backcall>=0.2.0",
        "bash_kernel>=0.9.3",
        "bleach>=4.1.0",
        "certifi>=2024.8.30",
        "cffi>=1.15.1",
        "charset-normalizer>=2.0.12",
        "comm>=0.1.4",
        "contextvars>=2.4",
        "corner>=2.2.1",
        "cycler>=0.11.0",
        "decorator>=5.1.1",
        "defusedxml>=0.7.1",
        "emcee>=3.1.6",
        "entrypoints>=0.4",
        "h5py>=3.1.0",
        "idna>=3.8",
        "immutables>=0.19",
        "importlib-metadata>=4.8.3",
        "importlib-resources>=5.4.0",
        "ipykernel>=5.5.6",
        "ipython>=7.16.3",
        "ipython-genutils>=0.2.0",
        "ipywidgets>=7.8.4",
        "jedi>=0.17.2",
        "Jinja2>=3.0.3",
        "joblib>=1.1.1",
        "json5>=0.9.16",
        "jsonschema>=3.2.0",
        "jupyter>=1.1.1",
        "jupyter-client>=7.1.2",
        "jupyter-console>=6.4.3",
        "jupyter-core>=4.9.2",
        "jupyter-server>=1.13.1",
        "jupyterlab>=3.2.9",
        "jupyterlab-pygments>=0.1.2",
        "jupyterlab-server>=2.10.3",
        "jupyterlab_widgets>=1.1.10",
        "kiwisolver>=1.3.1",
        "MarkupSafe>=2.0.1",
        "matplotlib>=3.3.4",
        "mistune>=0.8.4",
        "nbclassic>=0.3.5",
        "nbclient>=0.5.9",
        "nbconvert>=6.0.7",
        "nbformat>=5.1.3",
        "nest-asyncio>=1.6.0",
        "notebook>=6.4.10",
        "numpy>=1.19.5",
        "packaging>=21.3",
        "pandas>=1.1.5",
        "pandocfilters>=1.5.1",
        "parso>=0.7.1",
        "pexpect>=4.9.0",
        "pickleshare>=0.7.5",
        "Pillow>=8.4.0",
        "prometheus-client>=0.17.1",
        "prompt-toolkit>=3.0.36",
        "ptyprocess>=0.7.0",
        "pycparser>=2.21",
        "Pygments>=2.14.0",
        "pyparsing>=3.0.7",
        "pyrsistent>=0.18.0",
        "python-dateutil>=2.9.0.post0",
        "pytz>=2024.1",
        "pyzmq>=25.1.2",
        "requests>=2.27.1",
        "scipy>=1.5.4",
        "Send2Trash>=1.8.3",
        "setuptools>=59.6.0",
        "setuptools-scm>=6.4.2",
        "six>=1.16.0",
        "sniffio>=1.2.0",
        "terminado>=0.12.1",
        "testpath>=0.6.0",
        "tomli>=1.2.3",
        "tornado>=6.1",
        "tqdm>=4.64.1",
        "traitlets>=4.3.3",
        "typing_extensions>=4.1.1",
        "urllib3>=1.26.20",
        "wcwidth>=0.2.13",
        "webencodings>=0.5.1",
        "websocket-client>=1.3.1",
        "widgetsnbextension>=3.6.9",
        "zipp>=3.6.0"
    ],
    project_urls={  # Optional
        "Bug Reports": "https://github.com/federicoesposito/galaxySLED/issues",
        "Source": "https://github.com/federicoesposito/galaxySLED",
    },
)
