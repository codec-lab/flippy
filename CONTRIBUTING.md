# Install dependencies

There are several dependencies for development. Install them with
```
pip install '.[dev]'
```

To run the notebooks you might need the following:
```
pip install -e '.[dev]'
```

# Development process

Development happens in the `flippy-dev` repository, where `main` is the primary branch used for development, while `production` is used for releases. The public `flippy` repository has a `main` branch that mirrors the `production` branch in `flippy-dev`.

The release process can be run with
```bash
make release_dev_main
```

If using https-based authentication for git you can also try
```bash
make release_dev_main PRODUCTION_REPO='https://github.com/codec-lab/flippy-dev.git'
```

Before releasing, you will probably want to update the version in `version.txt` (the single place the version number is maintained) and commit the change.

# For a one-off change to production

Make sure you're on main when you start. This assumes `origin` is the dev repo, just like `make release_dev_main`.

```bash
git checkout -B production
git push origin production
git push git@github.com:codec-lab/flippy.git production:main
```

# Publishing to pypi

```bash
rm -r dist/*
python -m build
python -m twine upload --repository pypi dist/*
```
