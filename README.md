# Subsurface DA with Generative Models

This is a simple template for a collection of python scripts with testing (not a package).  
It uses pip for installation, flake8 for linting, pytest for testing, and coverage for monitoring test coverage.

To use it, first create a virtual environment, and install flake8, pytest, and coverage using pip.  
The following works on Windows: 
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then, install the dependencies:
```
pip install -r requirements.txt
pip install -e .
```

When done, deactivate the virtual environment:
```
deactivate
```
