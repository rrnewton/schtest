# CPU Scheduling Experiment Makefile
# ===================================

# Configuration
PYTHON := ../.venv/bin/python
VENV_DIR := ../.venv
PYTHON_DEPS := pandas matplotlib numpy pyyaml mypy pyright types-PyYAML pandas-stubs
RESULTS_DIR := results
EXPERIMENT_SCRIPT := mem_balance.py
ANALYSIS_SCRIPT := analyze_results.py
PLOT_FILE := $(RESULTS_DIR)/experiment_results.png
CSV_FILE := $(RESULTS_DIR)/experiment_results.csv

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

.PHONY: install_deps
install_deps: ## Install Python virtual environment and dependencies
	@echo "📦 Setting up Python environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment at $(VENV_DIR)..."; \
		python3 -m venv $(VENV_DIR); \
	fi
	@echo "Installing Python packages: $(PYTHON_DEPS)"
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install $(PYTHON_DEPS)
	@echo "✅ Python environment ready at $(VENV_DIR)"

.PHONY: full_run
full_run: check_deps clean experiment analyze ## Run complete experiment: setup → experiment → analysis
	@echo ""
	@echo "🎉 Full experiment completed successfully!"
	@echo "📊 Results: $(CSV_FILE)"
	@echo "📈 Plot: $(PLOT_FILE)"
	@echo ""

.PHONY: experiment
experiment: $(CSV_FILE) ## Run the CPU scheduling experiment only

$(CSV_FILE): $(EXPERIMENT_SCRIPT)
	@echo "🧪 Running CPU scheduling experiment..."
	@echo "⏱️  This will take approximately 90 seconds..."
	$(PYTHON) $(EXPERIMENT_SCRIPT)

.PHONY: analyze
analyze: $(CSV_FILE) ## Run detailed analysis of existing results
	@echo "📊 Running detailed analysis..."
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
	cd .. && .venv/bin/python -m pyright scripts/$(EXPERIMENT_SCRIPT) scripts/$(ANALYSIS_SCRIPT) scripts/test_experiment.py

.PHONY: typecheck-mypy
typecheck-mypy: ## Run only mypy type checking
	@echo "🔍 Running mypy type checking..."
	$(PYTHON) -m mypy --strict $(EXPERIMENT_SCRIPT) $(ANALYSIS_SCRIPT) test_experiment.py

.PHONY: check_deps
check_deps: ## Check dependencies
	@echo "🔍 Checking dependencies..."
	@if [ ! -f "$(PYTHON)" ]; then \
		echo "❌ Virtual environment not found. Run 'make install_deps' first."; \
		exit 1; \
	fi
	@$(PYTHON) --version
	@which stress-ng > /dev/null || (echo "❌ stress-ng missing (install with: sudo dnf install stress-ng)"; exit 1)
	@which perf > /dev/null || (echo "❌ perf missing (install with: sudo dnf install perf)"; exit 1)
	@$(PYTHON) -c "import pandas, matplotlib, numpy, yaml, mypy" 2>/dev/null || (echo "❌ Python packages missing. Run 'make install_deps'"; exit 1)
	@echo "✅ All dependencies OK!"

.PHONY: clean
clean: ## Clean up all generated files
	@echo "🧹 Cleaning up..."
	rm -f stress.sh metrics_*.yaml
	rm -rf $(RESULTS_DIR) __pycache__
	@echo "✅ Cleanup complete!"

.PHONY: view_results
view_results: $(PLOT_FILE) ## View the results plot
	xdg-open $(PLOT_FILE) 2>/dev/null || echo "Plot: $(PLOT_FILE)"

.PHONY: system_info
system_info: ## Show system information
	@echo "🖥️  System: $$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l) cores"
	@echo "Memory: $$(free -h | awk '/^Mem:/ {print $$2}')"
