.PHONY: install test lint train submit clean

install:
	pip install -e ".[dev,notebooks]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

download-data:
	kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/
	python -c "import zipfile; zipfile.ZipFile('data/raw/store-sales-time-series-forecasting.zip').extractall('data/raw/')"

train:
	python -m src.pipeline --train

submit:
	python -m src.pipeline --submit

pipeline:
	python -m src.pipeline

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
