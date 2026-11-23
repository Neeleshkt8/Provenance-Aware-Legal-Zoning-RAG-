install:
	pip install -r requirements.txt
serve:
	uvicorn zoning_rag.api:app --reload --port 8000
test:
	pytest -q
eval:
	python -m eval.run_eval --gate
eval-offline:
	python -m eval.run_eval --provider extractive --no-judge --gate
