.PHONY: test gen train eval

test:
	pytest -q

gen:
	python scripts/generate_dataset.py --config configs/small_debug.yaml

train:
	python scripts/train_cnn.py --config configs/small_debug.yaml

eval:
	python scripts/evaluate_model.py --config configs/small_debug.yaml
