autobuild:
	sphinx-autobuild -b html ./docs ./docs/_build/html

thesis:
	python setup.py sdist
	pip install dist/pdsm-0.1.0.tar.gz
	thesis_test