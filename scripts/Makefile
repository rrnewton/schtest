# CPU Scheduling Experiment Makefile
# ===================================

# Configuration
PYTHON := ../.venv/bin/python
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
	@$(PYTHON) --version
	@which stress-ng > /dev/null || (echo "❌ stress-ng missing"; exit 1)
	@which perf > /dev/null || (echo "❌ perf missing"; exit 1)
	@$(PYTHON) -c "import pandas, matplotlib, numpy, yaml" 2>/dev/null || (echo "❌ Python packages missing"; exit 1)
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
