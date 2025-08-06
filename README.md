<h1 align="center">
  LungUtils 🐍🩻
</h1>

<div align="center">

[![Pipeline](https://github.com/maxiludwig/lung_utils/actions/workflows/main_pipeline.yml/badge.svg)](https://github.com/maxiludwig/lung_utils/actions/workflows/main_pipeline.yml)
[![Documentation](https://github.com/maxiludwig/lung_utils/actions/workflows/main_documentation.yml/badge.svg)](https://maxiludwig.github.io/lung_utils/)
[![Coverage badge](https://github.com/maxiludwig/lung_utils/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/maxiludwig/lung_utils/tree/python-coverage-comment-action-data)

</div>

LungUtils is a quick-start Python repository to act as a skeleton for various projects around the multiphysics research code [4C](https://www.4c-multiphysics.org/) and leverages utilities from [PyToDa](https://github.com/davidrudlstorfer/pytoda). It includes the following basic amenities and tools:

- [PyTest](https://docs.pytest.org/) testing framework including an enforced minimum coverage check
- Automated [Github CI/CD](https://resources.github.com/devops/ci-cd/)
- Exhaustive [Pre-Commit](https://pre-commit.com) framework to automatically check code formatting and code quality
- Automatically generated [Documentation](https://pdoc.dev) based on the included Python docstrings
- Pre-defined framework to gather global settings (see [`main_example_config.yaml`](./src/lung_utils/main_example_config.yaml)) and execute a specific workflow

The remaining parts of the readme are structured as follows:

- [Setup](#setup)
- [Installation](#installation)
- [Execution](#execution)
  - [Execute LungUtils](#execute-lungutils)
  - [Run testing framework and create coverage report](#run-testing-framework-and-create-coverage-report)
  - [Create documentation](#create-documentation)
- [Dependency Management](#dependency-management)
- [Contributing](#contributing)
- [License](#license)


## Setup

To setup a project based on LungUtils simply follow these steps:

1. On Github create a new repository with `Use this template` and `Create a new repository`.
2. Clone your new repository to your local machine and setup the project according to the installation procedure down below.
3. Rename all occurrences of LungUtils to your new code project's name.
4. Within your GitHub project's settings, enable GitHub Pages for GitHub Actions (`Settings` -> `Pages` -> `Source` = `GitHub Actions`).
5. Create a new branch protection rule for the main branch on GitHub (`Settings` -> `Branches` -> `Add rule` -> Set up your rules`).
6. Activate to automatically delete branches after merging (`Settings` -> `General` -> `Automatically delete head branches`)

## Installation

For a quick and easy start an Anaconda/Miniconda environment is highly recommended. Other ways to install LungUtils are possible but here the installation procedure is explained based on a conda install. After installing Anaconda/Miniconda
execute the following steps:

- Create a new Anaconda environment based on the [`environment.yml`](./environment.yml) file:
```
conda env create -f environment.yml
```

- Activate your newly created environment:
```
conda activate lung_utils
```

- Initialize all submodules
```
git submodule update --init --recursive
```

- All necessary third party libraries for all submodules can be installed using:
```
git submodule --quiet foreach --recursive pip install -e .
```

- Install all LungUtils requirements with:
```
pip install -e .
```

- Finally, install the pre-commit hook with:
```
pre-commit install
```

Now you are up and running 🎉

## Execution

### Execute LungUtils

To execute LungUtils either run

```
lung_utils
````

to execute LungUtils with the provided exemplary config or use

```
lung_utils --config_file_path ../path/to/config.yaml
````

to utilize your own externally provided config file. Therein, all necessary configurations can be found.

### Run testing framework and create coverage report

To locally execute the tests and create the html coverage report simply run

```
pytest
```

### Create documentation

To locally create the documentation from the provided docstrings simply run

```
pdoc --html --output-dir docs src/lung_utils
```

## Dependency Management

To ease the dependency update process [`pip-tools`](https://github.com/jazzband/pip-tools) is utilized. To create the necessary [`requirements.txt`](./requirements.txt) file simply execute

```
pip-compile --all-extras --output-file=requirements.txt requirements.in
````

To upgrade the dependencies simply execute

```
pip-compile --all-extras --output-file=requirements.txt --upgrade requirements.in
````

Finally, perforfmance critical packages such as Numpy and Numba are installed via conda to utilize BLAS libraries.

## Contributing

All contributions are welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for more information.

## License

This project is licensed under a MIT license. For further information check [`LICENSE.md`](./LICENSE.md).
