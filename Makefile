\
# Convenience targets for the roof segmentation pipeline.
# On Windows, run via a shell that has `make` (e.g. Git Bash, MSYS2, WSL).

VENV_DIR   := .venv
PYTHON     := $(VENV_DIR)/Scripts/python.exe
PIP        := $(PYTHON) -m pip

SEED       ?= 42
EPOCHS     ?= 50
CHECKPOINT ?= outputs/checkpoints/best_model.pt

.PHONY: help venv install inspect data-convert check-dataset train predict test clean distclean

help:
	@echo "Targets:"
	@echo "  make venv           create virtualenv at $(VENV_DIR)"
	@echo "  make install        install dependencies + roof_seg package into $(VENV_DIR)"
	@echo "  make inspect        run dataset inspection on data/data_org/ (SPEC 3.2)"
	@echo "  make data-convert   (re)build data/data_convert/ (RGB images + label>128 labels) from data/data_org/"
	@echo "  make check-dataset  sanity-check the dataset loader (SPEC 3.3) + overlay figure"
	@echo "  make train          run training (SPEC 3.6); EPOCHS=$(EPOCHS) SEED=$(SEED)"
	@echo "  make predict        run inference on test images (SPEC 3.8); CHECKPOINT=$(CHECKPOINT)"
	@echo "  make test           run the test suite with pytest"
	@echo "  make clean          remove generated outputs (checkpoints, predictions, inspection)"
	@echo "  make distclean      clean + remove the virtualenv"

$(PYTHON):
	py -3.11 -m venv $(VENV_DIR)

venv: $(PYTHON)

install: venv
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt --no-warn-script-location -q
	$(PIP) install -e . --no-warn-script-location -q

inspect: venv
	$(PYTHON) scripts/inspect_data.py --seed $(SEED)

data-convert: venv
	$(PYTHON) scripts/build_data_convert.py

check-dataset: venv
	$(PYTHON) scripts/check_dataset.py --seed $(SEED)

train: venv
	$(PYTHON) scripts/train.py --seed $(SEED) --epochs $(EPOCHS)

predict: venv
	$(PYTHON) scripts/predict.py --seed $(SEED) --checkpoint $(CHECKPOINT)

test: venv
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf outputs/checkpoints/* outputs/predictions/* outputs/inspection/*
	rm -rf .pytest_cache roof_seg.egg-info
	find . -type d -name "__pycache__" -not -path "./$(VENV_DIR)/*" -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV_DIR)
