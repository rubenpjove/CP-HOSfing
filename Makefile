# Variables (you can export them beforehand if you prefer)
PROJECT := $(shell basename $(CURDIR))
EXP     ?= 
COMMENT ?= 
VENV    := $(STORE)/projects/$(PROJECT)/venv

MAKEFLAGS += --no-print-directory

venv:
	@module load cesga/2022 python/3.10.8 2>/dev/null || true; \
	if [ ! -d "$(VENV)/bin" ]; then \
	  python3 -m venv "$(VENV)"; \
	fi; \
	. "$(VENV)/bin/activate"; \
	pip install -U pip wheel; \
	test -f conf/requirements.txt && pip install -r conf/requirements.txt || true
	@echo "[OK] venv ready at $(VENV)"

# Target for run - handles both listing and execution
run:
	@if [ -z "$(filter-out run,$(MAKECMDGOALS))" ]; then \
		echo "[i] Available experiments:"; \
		ls -1 exps/ 2>/dev/null | sed 's/^/  - /' || echo "  (none found)"; \
		echo ""; \
		echo "[i] Usage: make run <experiment_name> [COMMENT=\"comment\"]"; \
		echo "[i] Example: make run baseA or make run exp1/v1 COMMENT=\"this is a comment\""; \
	else \
		EXPERIMENT_PATH="$(filter-out run,$(MAKECMDGOALS))"; \
		FULL_PATH="$(PROJECT)/$$EXPERIMENT_PATH"; \
		MAIN_PY_PATH="exps/$$EXPERIMENT_PATH/src/main.py"; \
		echo "[i] Running project $$FULL_PATH"; \
		echo "[i] Checking main.py exists..."; \
		if [ ! -f "$$MAIN_PY_PATH" ]; then \
			echo "[ERROR] main.py not found at: $$MAIN_PY_PATH"; \
			echo "[ERROR] Check that the experiment path '$$EXPERIMENT_PATH' is correct"; \
			echo "[ERROR] Expected structure: exps/$$EXPERIMENT_PATH/src/main.py"; \
			exit 1; \
		fi; \
		echo "[i] main.py found at: $$MAIN_PY_PATH"; \
		echo "[i] Generating and submitting job..."; \
		. "$(VENV)/bin/activate"; \
		python tools/exp.py --path "$$FULL_PATH" --submit $(if $(COMMENT),--comment "$(COMMENT)"); \
		echo "[OK] Submitted."; \
	fi

# Prevent make from trying to build the experiment names as targets
%:
	@:

go: venv
	@$(MAKE) run $(filter-out go,$(MAKECMDGOALS))

# Example sweep (job array)
# sweep: venv setup_dirs
# 	@. "$(VENV)/bin/activate"; \
# 	python tools/exp.py --project "$(PROJECT)" --set hyp.alpha=0.2 --submit --array 0-9

# Clean a run (does not delete artifacts in STORE)
# clean-last:
# 	@echo "Implement if you want: delete the last run in LUSTRE"

.PHONY: venv run go #sweep clean-last
