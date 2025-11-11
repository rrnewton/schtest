#!/bin/bash
# Setup passwordless sudo for scheduler experiment commands
#
# This script creates a sudoers configuration file that allows
# running dmesg and the scx_lavd scheduler without a password.
#
# Usage: sudo ./setup_sudo.sh

set -e

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo"
    echo "Usage: sudo $0"
    exit 1
fi

SUDOERS_DIR="/etc/sudoers.d"
SUDOERS_FILE="$SUDOERS_DIR/schtest"
USERNAME="${SUDO_USER:-$USER}"

echo "Setting up passwordless sudo for user: $USERNAME"

# Ensure /etc/sudoers.d directory exists
if [ ! -d "$SUDOERS_DIR" ]; then
    echo "Creating $SUDOERS_DIR directory..."
    mkdir -p "$SUDOERS_DIR"
    chmod 0750 "$SUDOERS_DIR"
fi

# Find the scx_lavd binary (optional)
SCX_LAVD_PATH="$(realpath ../../scx/target/release/scx_lavd 2>/dev/null || echo "")"

if [ -z "$SCX_LAVD_PATH" ] || [ ! -f "$SCX_LAVD_PATH" ]; then
    echo "Note: scx_lavd binary not found at ../../scx/target/release/scx_lavd"
    echo "Only dmesg will be configured for passwordless sudo."
    echo "(You can run this script again after building the scheduler)"
    SCX_LAVD_PATH=""
else
    echo "Found scheduler at: $SCX_LAVD_PATH"
fi

# Create sudoers configuration
cat > "$SUDOERS_FILE" << EOF
# Passwordless sudo for schtest experiment commands
# Created by setup_sudo.sh on $(date)

# Allow dmesg for monitoring scheduler messages
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/dmesg
EOF

# Add scheduler line only if we found it
if [ -n "$SCX_LAVD_PATH" ]; then
    cat >> "$SUDOERS_FILE" << EOF

# Allow running the scx_lavd scheduler
$USERNAME ALL=(ALL) NOPASSWD: $SCX_LAVD_PATH
EOF
fi

# Set proper permissions (sudoers files must be 0440)
chmod 0440 "$SUDOERS_FILE"

# Validate the sudoers file
if visudo -c -f "$SUDOERS_FILE" >/dev/null 2>&1; then
    echo "✅ Sudoers configuration created successfully: $SUDOERS_FILE"
    echo ""
    echo "The following commands can now run without password:"
    echo "  - sudo dmesg"
    if [ -n "$SCX_LAVD_PATH" ]; then
        echo "  - sudo $SCX_LAVD_PATH"
    fi
    echo ""
    echo "You can now run: ./mem_balance.py"
else
    echo "❌ Error: Invalid sudoers configuration"
    rm -f "$SUDOERS_FILE"
    exit 1
fi
