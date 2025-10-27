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

SUDOERS_FILE="/etc/sudoers.d/schtest"
USERNAME="${SUDO_USER:-$USER}"

echo "Setting up passwordless sudo for user: $USERNAME"

# Find the scx_lavd binary
SCX_LAVD_PATH="$(realpath ../../scx/target/release/scx_lavd 2>/dev/null || echo "")"

if [ -z "$SCX_LAVD_PATH" ] || [ ! -f "$SCX_LAVD_PATH" ]; then
    echo "Warning: scx_lavd binary not found at ../../scx/target/release/scx_lavd"
    echo "Please build the scheduler first with: cd ../../scx && cargo build --release"
    exit 1
fi

echo "Found scheduler at: $SCX_LAVD_PATH"

# Create sudoers configuration
cat > "$SUDOERS_FILE" << EOF
# Passwordless sudo for schtest experiment commands
# Created by setup_sudo.sh on $(date)

# Allow dmesg -W for monitoring scheduler messages
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/dmesg

# Allow running the scx_lavd scheduler
$USERNAME ALL=(ALL) NOPASSWD: $SCX_LAVD_PATH
EOF

# Set proper permissions (sudoers files must be 0440)
chmod 0440 "$SUDOERS_FILE"

# Validate the sudoers file
if visudo -c -f "$SUDOERS_FILE" >/dev/null 2>&1; then
    echo "✅ Sudoers configuration created successfully: $SUDOERS_FILE"
    echo ""
    echo "The following commands can now run without password:"
    echo "  - sudo dmesg"
    echo "  - sudo $SCX_LAVD_PATH"
    echo ""
    echo "You can now run: make experiment"
else
    echo "❌ Error: Invalid sudoers configuration"
    rm -f "$SUDOERS_FILE"
    exit 1
fi
