#!/bin/bash

# --- Configuration ---
SERVICE_FILE="/etc/systemd/system/LWBOT.service"
VARIABLE_NAME="NEW_TOKEN"
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

# --- Check for root/sudo privileges ---
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run with root privileges (sudo)."
    exit 1
fi

# --- 1. Edit the Service File ---
echo "Updating $VARIABLE_NAME in $SERVICE_FILE..."
sed -i "s#^Environment=\"$VARIABLE_NAME=.*#Environment=\"$VARIABLE_NAME=$NEW_TOKEN\"#g" "$SERVICE_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully updated $VARIABLE_NAME."
else
    echo "Error: Failed to update $VARIABLE_NAME."
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
    echo "Error: Failed to restart $SERVICE_NAME."
    exit 1
fi
