# Makefile Usage Guide

## Quick Start

The Makefile provides a convenient interface to run the CPU scheduling experiments. Here are the most important commands:

### Run Full Experiment
```bash
make full_run
```
This will:
1. Check all dependencies are available
2. Clean up any previous results
3. Run the complete experiment (takes ~90 seconds)
4. Generate detailed analysis
5. Create performance plots

### Available Targets

| Target | Description |
|--------|-------------|
| `make help` | Show all available targets with descriptions |
| `make full_run` | Run complete experiment with analysis |
| `make experiment` | Run only the experiment (no analysis) |
| `make analyze` | Analyze existing results |
| `make quick_test` | Run a 2-second test to verify setup |
| `make check_deps` | Verify all dependencies are installed |
| `make clean` | Remove all generated files |
| `make view_results` | Open the results plot |
| `make system_info` | Show system specs relevant to experiment |

## Dependencies

The experiment requires:
- **stress-ng**: CPU and memory stress testing tool
- **perf**: Linux performance monitoring tool  
- **Python packages**: Managed via pip-tools (see [DEPENDENCIES.md](../DEPENDENCIES.md))

### System Dependencies
Install system tools:
```bash
# On Ubuntu/Debian:
sudo apt-get install stress-ng linux-tools-generic

# On Fedora/RHEL:
sudo dnf install stress-ng perf
```

### Python Dependencies  
Install Python packages from lock file:
```bash
make install_deps        # Production dependencies
make install_deps_dev    # Development dependencies
```

Check all dependencies:
```bash
make check_deps
```

For detailed dependency management, see [DEPENDENCIES.md](../DEPENDENCIES.md).

## Example Workflow

1. **First time setup**: 
   ```bash
   make check_deps  # Verify everything is ready
   ```

2. **Run experiment**: 
   ```bash
   make full_run    # Complete experiment with analysis
   ```

3. **View results**: 
   ```bash
   make view_results  # Open the performance plot
   ```

4. **Re-run later**:
   ```bash
   make clean       # Clean up old results
   make full_run    # Run fresh experiment
   ```

## Output Files

After running the experiment, you'll find:

- `results/experiment_results.csv` - Raw performance data
- `results/experiment_results.png` - Performance visualization
- `results/perf_*.json` - Detailed perf counter data
- Various `metrics_*.yaml` files - stress-ng output

## Tips

- The full experiment takes about 90 seconds (9 configurations × 10 seconds each)
- Use `make quick_test` to verify setup before running the full experiment
- Results are automatically saved with timestamps for comparison
- The experiment tests different thread pinning strategies: none, spread, and half

## Troubleshooting

If you get permission errors with `perf`, you may need to adjust system settings:
```bash
# Allow perf for non-root users (temporary)
sudo sysctl kernel.perf_event_paranoid=1
```

For more details on the experiment design, see `RESULTS.md`.