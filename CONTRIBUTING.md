# Contributing to FCEst

This file is still a work in progress.

This file contains notes for potential contributors to FCEst, as well as some notes that may be helpful for maintenance.

#### Table Of Contents

* [Project scope](#project-scope)
* [Code quality requirements](#code-quality-requirements)
* [Pull requests and the master branch](#pull-requests-and-the-master-branch)
* [Tests and continuous integration](#tests-and-continuous-integration)

## Project scope

FCEst is meant to be a comprehensive library that can take care of all tasks involved in estimation of functional connectivity.
This is why it includes an extensive list of preferred estimation methods.

We have aimed for the high-level API for all estimation methods to be similar.

We welcome contributions to FCEst.
If you would like to contribute a feature, please raise discussion via a GitHub issue, to discuss the suitability of the feature within FCEst.
If the feature is outside the envisaged scope, we can still link to a separate project in our Readme.

## Code quality requirements

- Code must be covered by tests. This is still a work in progress. We strongly encourage you to use the [pytest](https://docs.pytest.org/) framework.
- The code must be documented. We use *reST* in docstrings. *reST* is a [standard way of documenting](http://docs.python-guide.org/en/latest/writing/documentation/) in python.\
If the code which you are working on does not yet have any documentation, we would be very grateful if you could amend the deficiency.
Missing documentation leads to ambiguities and difficulties in understanding future contributions and use cases.
- Use [type annotations](https://docs.python.org/3/library/typing.html). Type hints make code cleaner and _safer_ to some extent.
- Python code should generally follow the *PEP8* style.
- Practise writing good code as far as is reasonable. Simpler is usually better. Reading the existing FCEst code should give a good idea of the expected style.

### Naming conventions

Variable names: scalars and vectors start lowercase, but following the notation used in Gaussian process papers, all matrices are denoted with upper case.

### Formatting

FCEst uses [black](https://github.com/psf/black) and [isort](https://pycqa.github.io/isort/) for formatting.
Simply run `make format` from the FCEst root directory (or check our Makefile for the appropriate command-line options).

## Pull requests

If you think that your contribution falls within the project scope (see above) please submit a Pull Request (PR) to our GitHub page.
(GitHub provides extensive documentation on [forking](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) and [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).)

In order to maintain code quality, and make life easy for the reviewers, please ensure that your PR:

- Only fixes one issue or adds one feature.
- Makes the minimal amount of changes to the existing codebase.
- Is testing its changes.
- Passes all checks (formatting, types, tests - you can run them all locally using `make check-all` from the FCEst root directory).

All code goes through a PR; there are no direct commits to the main branch.

## Tests and continuous integration

FCEst is not yet fully covered by the testing suite. We expect changes to code to pass these tests, and for new code to be covered by new tests.

## Documentation

FCEst does not have a documentation yet.

## Version numbering

The main purpose of versioning FCEst is user convenience.

We use the [semantic versioning scheme](https://semver.org/).
The semver implies `MAJOR.MINOR.PATCH` version scheme, where `MAJOR` changes when there are incompatibilities in API, `MINOR` means adding functionality without breaking existing API and `PATCH` presumes the code update has backward compatible bug fixes.
