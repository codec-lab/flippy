test:
	python -m py.test $(ARGS)

lint:
	# Copied from .github/workflows
	flake8 gorgo --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 gorgo --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

docs:
	pdoc --math gorgo

docs_build:
	pdoc --math gorgo -o docs
	$(MAKE) tutorials_build

tutorials_build:
	for f in tutorials/*.ipynb; do \
		jupyter nbconvert --to markdown $$f; \
    done
