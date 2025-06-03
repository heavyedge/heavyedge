# HeavyEdge

Basic package for heavy edge analysis.

Provides:

- Read and process edge profile data for analysis.
- Utility commands for data management.

## Documentation

Package is documented with [Sphinx](https://pypi.org/project/Sphinx/).
If you want to build the document yourself, get the source code and install with `[doc]` dependency.
Then, go to `doc` directory and build the document:

```
$ pip install .[doc]
$ cd doc
$ make html
```

Document will be generated in `build/html` directory. Open `index.html` to see the central page.

## Developing

### Installation

For development features, you must install the package by `pip install -e .[doc]`.

### Testing

Run `pytest` command to perform unit test.

When doctest is run, buildable sample data are rebuilt by default.
To disable this, set `HEAVYEDGE_TEST_REBUILD` environment variable to zero.
For example,
```
HEAVYEDGE_TEST_REBUILD=0 pytest
```
