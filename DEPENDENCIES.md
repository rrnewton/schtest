# Dependency Management

This project uses **pip-tools** for deterministic dependency management, similar to Cargo.lock in Rust or package-lock.json in Node.js.

## Files

- `requirements.in` - High-level dependencies (what you want)
- `requirements.txt` - Lock file with exact versions (what you get)

## Workflows

### Development Setup
```bash
# Install all dependencies from lock file
make install_deps
```

### Adding New Dependencies
1. Add to `requirements.in`
2. Update lock file: `make update_deps`
3. Install new dependencies: `make install_deps`

### Development Dependencies
```bash
# Install deps + development tools (editable mode)
make install_deps_dev
```

### Updating Dependencies
```bash
# Regenerate lock file with latest compatible versions
make update_deps

# Install updated dependencies
make install_deps
```

## Why Lock Files?

Lock files ensure:
- **Reproducible builds** - Same versions across environments
- **Security** - Known good versions, audit trail
- **Stability** - Prevents surprise updates breaking your code
- **Team collaboration** - Everyone uses identical dependencies

## Makefile Targets

- `install_deps` - Install from lock file (production)
- `install_deps_dev` - Install with development tools
- `update_deps` - Regenerate lock file from .in file
- `check_deps` - Verify dependencies are in sync
- `typecheck` - Run type checking (requires dependencies)
- `full_run` - Complete experiment workflow

## Best Practices

1. **Always commit both files** - `requirements.in` AND `requirements.txt`
2. **Use specific versions in .in** only when necessary (security fixes, etc.)
3. **Update regularly** - `make update_deps` to get security updates
4. **Never edit .txt directly** - Always update via .in file