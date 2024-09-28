.PHONY: install run

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

run:
	FLASK_APP=app.py FLASK_RUN_HOST=localhost FLASK_RUN_PORT=3000 flask run
