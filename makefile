version = $(shell cat package.json | grep version | awk -F'"' '{print $$4}')

install:
	pip3 install poetry black isort
	poetry install

run:
	poetry run python3 main.py

lint:
	isort main.py
	black main.py
