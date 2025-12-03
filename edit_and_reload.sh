cat edit_and_reload.sh 
#!/bin/bash

# --- Configuration ---
# Set the path to the systemd service file you want to edit
SERVICE_FILE="/etc/systemd/system/LWBOT.service"
# Set the name of the variable to be updated (e.g., 'NEW_TOKEN')
VARIABLE_NAME="NEW_TOKEN"
# Set the name of the systemd service
SERVICE_NAME="LWBOT.service"

# --- Input Validation ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <New_Token_Value>"
    exit 1
fi

NEW_TOKEN="$1"

# --- Check if the service file exists ---
if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file not found at $SERVICE_FILE"
    exit 1
fi

# --- Check for root/sudo privileges (required for systemd and /etc editing) ---
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with root privileges (sudo)."
    exit 1
fi

# --- 1. Edit the Service File ---
echo "Updating $VARIABLE_NAME in $SERVICE_FILE..."
# The 'sed' command safely finds the line starting with 'Environment="VARIABLE_NAME=' and replaces the entire line
# The pattern uses a forward slash, so we use a different delimiter like '#' for the substitution to avoid escaping issues
# Note: The new token is treated as a literal string within the quotes for the environment variable.
sed -i "s#^Environment=\"$VARIABLE_NAME=.*#Environment=\"$VARIABLE_NAME=$NEW_TOKEN\"#g" "$SERVICE_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully updated $VARIABLE_NAME."
else
    echo "Error: Failed to update $VARIABLE_NAME. Check if the line exists and the file is correct."
    exit 1
fi

# --- 2. Reload the systemd Daemon ---
echo "Reloading systemd daemon..."
systemctl daemon-reload

if [ $? -ne 0 ]; then
    echo "Error: Failed to reload systemd daemon."
    exit 1
fi

# --- 3. Restart the Service ---
echo "Restarting $SERVICE_NAME to apply changes..."
systemctl restart "$SERVICE_NAME"

if [ $? -eq 0 ]; then
    echo "✅ Success! $SERVICE_NAME restarted with the new token."
else
    echo "Error: Failed to restart $SERVICE_NAME. Check service status with 'systemctl status $SERVICE_NAME'"
    exit 1
fi
randoo_online@hushh-online-vm:~$ 
