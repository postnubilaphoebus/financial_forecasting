#!/bin/bash

# Update pip and pipenv
pip install --upgrade pip
pip install pipenv

# Activate the pipenv environment or create a new one
python -m venv .venv
source .venv/bin/activate

# Install required packages from requirements.txt
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate