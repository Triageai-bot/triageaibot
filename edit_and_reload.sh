#!/bin/bash

# --- Configuration ---
SERVICE_FILE="/etc/systemd/system/LWBOT.service"
SERVICE_NAME="LWBOT.service"

# --- Input Validation ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <New_Token_Value> <WhatsApp_Phone_ID>"
    exit 1
fi

NEW_TOKEN="$1"
WHATSAPP_PHONE_ID="$2"

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

# --- Update NEW_TOKEN ---
echo "Updating NEW_TOKEN in $SERVICE_FILE..."
sed -i "s#^Environment=\"NEW_TOKEN=.*#Environment=\"NEW_TOKEN=$NEW_TOKEN\"#g" "$SERVICE_FILE"

# --- Update WHATSAPP_PHONE_ID ---
echo "Updating WHATSAPP_PHONE_ID in $SERVICE_FILE..."
sed -i "s#^Environment=\"WHATSAPP_PHONE_ID=.*#Environment=\"WHATSAPP_PHONE_ID=$WHATSAPP_PHONE_ID\"#g" "$SERVICE_FILE"

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Restarting $SERVICE_NAME..."
systemctl restart "$SERVICE_NAME"

if [ $? -eq 0 ]; then
    echo "✅ Success! $SERVICE_NAME restarted with updated NEW_TOKEN and WHATSAPP_PHONE_ID."
else
    echo "❌ Error restarting service."
    exit 1
fi
