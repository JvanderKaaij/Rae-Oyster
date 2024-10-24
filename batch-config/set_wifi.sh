#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <SSID> <Password>"
    exit 1
fi

# Assign arguments to variables
SSID="$1"
PASSWORD="$2"

# Generate wpa_supplicant configuration for the given SSID and password
WPA_OUTPUT=$(wpa_passphrase "$SSID" "$PASSWORD")

# Extract the network block
NETWORK_BLOCK=$(echo "$WPA_OUTPUT" | sed -n '/network={/,/}/p')

# Check if the network block was generated successfully
if [ -z "$NETWORK_BLOCK" ]; then
    echo "Failed to generate wpa_supplicant configuration."
    exit 1
fi

# Backup the existing wpa_supplicant configuration file
cp /etc/wpa_supplicant.conf /etc/wpa_supplicant.conf.bak

# Replace the network block in /etc/wpa_supplicant.conf
sed -i '/network={/,/}/d' /etc/wpa_supplicant.conf
printf "%s\n" "$NETWORK_BLOCK" >> /etc/wpa_supplicant.conf

# Stop hostapd service
systemctl stop hostapd

# Kill any existing wpa_supplicant process
pkill wpa_supplicant

# Remove stale control interface file if it exists
if [ -e /var/run/wpa_supplicant/wlp1s0 ]; then
    rm /var/run/wpa_supplicant/wlp1s0
fi

# Start wpa_supplicant with the updated configuration
wpa_supplicant -B -i wlp1s0 -c /etc/wpa_supplicant.conf

# Restart systemd-networkd service
systemctl restart systemd-networkd

# Print success message
echo "Wi-Fi configuration updated and services restarted successfully."