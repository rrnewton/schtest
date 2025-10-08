# CPU Scheduling Experiment Makefile
# ===================================

# Configuration
PYTHON := ../.venv/bin/python
VENV_DIR := ../.venv
REQUIREMENTS_IN := requirements.in
REQUIREMENTS_LOCK := requirements.txt
RESULTS_DIR := results
EXPERIMENT_SCRIPT := mem_balance.py
ANALYSIS_SCRIPT := analyze_results.py
LATEST_RESULTS := $(RESULTS_DIR)/latest
PLOT_FILE := $(LATEST_RESULTS)/experiment_results.png
CSV_FILE := $(LATEST_RESULTS)/experiment_results.csv

# Default target
.PHONY: help
help: ## Show this help message
	@echo "CPU Scheduling Experiment Makefile"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Machine name options:"
	@echo "  Direct execution:     $(PYTHON) mem_balance.py --machine=workstation"
	@echo "  Results organized:    results/<machine_name>/"
	@echo "  Auto-detect from:     /proc/cpuinfo model name"
	@echo ""

.PHONY: install_deps
install_deps: ## Install Python virtual environment and dependencies from lock file
	@echo "📦 Setting up Python environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment at $(VENV_DIR)..."; \
		python3 -m venv $(VENV_DIR); \
	fi
	@echo "Installing pip-tools..."
	@$(VENV_DIR)/bin/pip install --upgrade pip pip-tools
	@if [ ! -f "$(REQUIREMENTS_LOCK)" ]; then \
		echo "📋 Lock file not found, generating from $(REQUIREMENTS_IN)..."; \
		$(MAKE) update_deps; \
	fi
	@echo "📦 Installing dependencies from lock file $(REQUIREMENTS_LOCK)..."
	@$(VENV_DIR)/bin/pip-sync $(REQUIREMENTS_LOCK)
	@echo "✅ Python environment ready at $(VENV_DIR)"

.PHONY: update_deps
update_deps: ## Update dependencies and regenerate lock file
	@echo "🔄 Updating dependencies from $(REQUIREMENTS_IN)..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "❌ Virtual environment not found. Run 'make install_deps' first."; \
		exit 1; \
	fi
	@$(VENV_DIR)/bin/pip-compile --upgrade --output-file $(REQUIREMENTS_LOCK) $(REQUIREMENTS_IN)
	@echo "✅ Lock file updated: $(REQUIREMENTS_LOCK)"

.PHONY: install_deps_dev
install_deps_dev: update_deps install_deps ## Update and install dependencies (for development)
	@echo "✅ Python environment ready at $(VENV_DIR)"

.PHONY: full_run
full_run: check_deps clean experiment analyze ## Run complete experiment: setup → experiment → analysis
	@echo ""
	@echo "🎉 Full experiment completed successfully!"
	@echo "📊 Results: $(CSV_FILE)"
	@echo "📈 Plot: $(PLOT_FILE)"
	@echo ""

.PHONY: experiment
experiment: check_deps ## Run the CPU scheduling experiment only
	@echo "🧪 Running CPU scheduling experiment..."
	$(PYTHON) $(EXPERIMENT_SCRIPT)

.PHONY: analyze
analyze: ## Run detailed analysis of existing results
	@echo "📊 Running detailed analysis..."
	@if [ ! -L "$(LATEST_RESULTS)" ]; then \
		echo "❌ No latest results found. Run 'make experiment' first."; \
		exit 1; \
	fi
	@if [ ! -f "$(CSV_FILE)" ]; then \
		echo "❌ Results CSV not found at $(CSV_FILE)"; \
		exit 1; \
	fi
	$(PYTHON) $(ANALYSIS_SCRIPT)

.PHONY: quick_test
quick_test: check_deps ## Run a quick 2-second test
	@echo "🚀 Running quick test..."
	$(PYTHON) test_experiment.py

.PHONY: typecheck
typecheck: typecheck-vscode typecheck-mypy

.PHONY: typecheck-vscode
typecheck-vscode: ## Run pyright type checking (matches VS Code exactly)
	@echo "🔍 Running pyright (VS Code type checker)..."
	$(PYTHON) -m pyright $(EXPERIMENT_SCRIPT) $(ANALYSIS_SCRIPT) test_experiment.py

.PHONY: typecheck-mypy
typecheck-mypy: ## Run only mypy type checking
	@echo "🔍 Running mypy type checking..."
	$(PYTHON) -m mypy --strict $(EXPERIMENT_SCRIPT) $(ANALYSIS_SCRIPT) test_experiment.py

.PHONY: check_deps
check_deps: ## Check if all dependencies are available
	@echo "🔍 Checking dependencies..."
	@which stress-ng > /dev/null || (echo "❌ stress-ng missing (install with: sudo dnf install stress-ng)"; exit 1)
	@which perf > /dev/null || (echo "❌ perf missing (install with: sudo dnf install perf)"; exit 1)
	@if [ -f "$(REQUIREMENTS_LOCK)" ]; then \
		echo "📦 Lock file found: $(REQUIREMENTS_LOCK)"; \
	else \
		echo "❌ Lock file missing. Run 'make update_deps' first"; exit 1; \
	fi
	@$(PYTHON) -c "import pkg_resources; pkg_resources.require(open('$(REQUIREMENTS_LOCK)').read().splitlines())" 2>/dev/null || \
		(echo "❌ Python packages not in sync with lock file. Run 'make install_deps'"; exit 1)
	@echo "✅ All dependencies OK!"

.PHONY: clean
clean: ## Clean up all generated files
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__ .mypy_cache
	@echo "✅ Cleanup complete!"

.PHONY: view_results
view_results: $(PLOT_FILE) ## View the results plot
	xdg-open $(PLOT_FILE) 2>/dev/null || echo "Plot: $(PLOT_FILE)"

.PHONY: system_info
system_info: ## Show system information
	@echo "🖥️  System: $$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l) cores"
	@echo "Memory: $$(free -h | awk '/^Mem:/ {print $$2}')"
