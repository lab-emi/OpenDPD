.SHELL := /bin/bash

.PHONY: help install uninstall clean clean-data clean-logs clean-models clean-all

INSTALL_CMD ?= pip install -e .
UNINSTALL_CMD ?= pip uninstall -y opendpd

help:
	@echo "OpenDPD Make Targets:"
	@echo "  make install         Install OpenDPD (editable). Override INSTALL_CMD if needed."
	@echo "  make uninstall       Uninstall OpenDPD. Override UNINSTALL_CMD if needed."
	@echo "  make clean           Remove generated logs, models, preds (safe cleanup)."
	@echo "  make clean-data      Remove generated dataset outputs (dpd_out/ run artifacts)."
	@echo "  make clean-logs      Remove log/ and bash_scripts logs."
	@echo "  make clean-models    Remove save/ artifacts."
	@echo "  make clean-all       Remove logs, models, dpd_out and caches."

install:
	@echo "[INFO] Installing OpenDPD..."
	@$(INSTALL_CMD)

uninstall:
	@echo "[INFO] Uninstalling OpenDPD..."
	@$(UNINSTALL_CMD)

clean-models:
	@echo "[INFO] Removing model checkpoints (save/)..."
	@chmod -R u+w save 2>/dev/null || true
	@rm -rf save 2>/dev/null || (echo "[WARN] Could not remove 'save'. Try removing manually if needed." >&2)

clean-logs:
	@echo "[INFO] Removing logs (log/)..."
	@chmod -R u+w log 2>/dev/null || true
	@rm -rf log 2>/dev/null || (echo "[WARN] Could not remove 'log'. Try removing manually if needed." >&2)

clean-data:
	@echo "[INFO] Removing generated DPD outputs (dpd_out/)..."
	@chmod -R u+w dpd_out 2>/dev/null || true
	@rm -rf dpd_out 2>/dev/null || (echo "[WARN] Could not remove 'dpd_out'. Try removing manually if needed." >&2)

clean-dist:
	@echo "[INFO] Removing distribution files (dist/)..."
	@chmod -R u+w dist 2>/dev/null || true
	@rm -rf dist 2>/dev/null || (echo "[WARN] Could not remove 'dist'. Try removing manually if needed." >&2)

clean-egg-info:
	@echo "[INFO] Removing egg-info files (*.egg-info/)..."
	@chmod -R u+w *.egg-info 2>/dev/null || true
	@rm -rf *.egg-info 2>/dev/null || (echo "[WARN] Could not remove '*.egg-info'. Try removing manually if needed." >&2)

clean-pycache:
	@echo "[INFO] Removing Python cache files (__pycache__/)..."
	@find . -name '__pycache__' -type d -exec rm -rf {} +
	@find . -name '*.pyc' -delete

clean-build:
	@echo "[INFO] Removing build files (build/)..."
	@chmod -R u+w build 2>/dev/null || true
	@rm -rf build 2>/dev/null || (echo "[WARN] Could not remove 'build'. Try removing manually if needed." >&2)

clean-all:
	@echo "[INFO] Removing all generated files..."
	@$(MAKE) clean-logs
	@$(MAKE) clean-models
	@$(MAKE) clean-data
	@$(MAKE) clean-dist
	@$(MAKE) clean-egg-info
	@$(MAKE) clean-pycache
	@$(MAKE) clean-build
	@echo "[INFO] Done."

clean:
	@$(MAKE) clean-all
	@echo "[INFO] Done."
