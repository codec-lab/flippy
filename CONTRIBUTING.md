# Install dependencies

There are several dependencies for development. Install them with
```
pip install '.[dev]'
```

# Development process

Development happens in the `flippy-dev` repository, where `main` is the primary branch used for development, while `production` is used for releases. The public `flippy` repository has a `main` branch that mirrors the `production` branch in `flippy-dev`.

The release process can be run with
```bash
make release_dev_main
```
