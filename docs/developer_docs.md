---
file_format: mystnb
kernelspec:
  name: python3
---

# Developer Documentation

## Setting up an Environment for NiAS Development

### Getting the Source

Clone the NiAS git repository using:

```shell
git clone git@github.com:nias-project/nias.git
cd nias
```

and, optionally, switch to the branch you are interested in, e.g.:

```shell
git checkout some_branch
```

### Environment with venv

Create and activate a new venv (alternatively, you can use
[`virtualenv`](https://virtualenv.pypa.io/en/latest/) and
[`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/)):

```shell
python3 -m venv venv
source venv/bin/activate
```

Then, it may be necessary to upgrade `pip`:

```shell
pip install -U pip
```

Finally, make an editable installation of NiAS with minimal dependencies:

```shell
pip install -e .[test,docs]
```

## Updating NiAS' Dependencies

All required or optional dependencies of NiAS are specified in `pyproject.toml`.

We use [pip-compile](https://github.com/jazzband/pip-tools), to generate `requirements-*.txt`
files from these specifications, which contain pinned versions of all required packages for the
CI runs.
The extras included into the images are specified in `Makefile`.

If you update NiAS' dependencies, make sure to execute

```shell
make ci_requirements
make docs_requirements
```

and commit the changes made to the lock files to ensure that the updated dependencies are picked up
by CI.

Note that these make targets require [docker](https://www.docker.com/) or a compatible
container runtime such as [podman](https://podman.io/).
The NiAS main developers will be happy to take care of this step for you.

(ref_testing_ci)=

## Testing / Continuous Integration Setup

NiAS uses [pytest](https://pytest.org/) for unit testing.
All tests are contained within the `tests` directory and can be run
by invoking the `pytest` executable.
Please refer to the [pytest documentation](https://docs.pytest.org/en/latest/how-to/usage.html)
for detailed examples.

## Creating a new release

To create a new release of NiAS, perform the following steps:

- Check that current TestPyPI version is working correctly by installing it with

  ```shell
  pip install --index-url https://test.pypi.org/simple/ \
              --extra-index-url https://pypi.org/simple/ \
              nias
  ```

- Bump version in `src/nias/__init__.py` to `a.b.c`.
- Create and merge PR with updated version.
- Tag release and push it with

  ```shell
  git tag va.b.c
  git push origin va.b.c
  ```

- Bump version in `src/nias/__init__.py` to `a.b.(c+1).dev0`.

Pushing the tag to GitHub will automatically trigger an upload to PyPI and create a GitHub release, which is also
uploaded to Zenodo.
