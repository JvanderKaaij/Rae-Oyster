#!/bin/bash

# Function to find rae-ID -> IP mappings
find_rae_ip_list() {
  # Associative array of devices: hostname -> MAC address
  declare -A devices=(
    [1_rae-4AD2201]="8C:1D:96:E2:4F:A5"
    [2_rae-3B12781]="F4:26:79:03:3D:47"
    [3_rae-3972A01]="60:DD:8E:BE:C9:04"
    [4_rae-49B2781]="60:DD:8E:BE:CB:52"
    [5_rae-3AF2A01]="F4:26:79:EA:84:5E"
    [6_rae-30F2D01]="F4:26:79:03:3D:65"
    [7_rae-3332801]="8C:1D:96:E2:4F:BE"
    [8_rae-3A12401]="60:DD:8E:A7:08:B4"
    [9_rae-6B52901]="F4:26:79:EA:85:17"
    [10_rae-7BF2881]="60:DD:8E:BE:AF:F5"
  )

  # Define the network range to scan
  local network_range="146.50.60.1-90"

  # Declare an associative array to store IP and MAC address pairs
  declare -A ip_mac_list

  # Use process substitution to avoid the subshell issue
  while read -r line; do
      if [[ "$line" =~ "Nmap scan report for" ]]; then
          # Extract the IP address and remove parentheses if present
          ip=$(echo "$line" | awk '{print $NF}' | sed 's/[()]//g')
          read -r host_status
          read -r mac_line
          if [[ "$mac_line" =~ "MAC Address" ]]; then
            mac=$(echo "$mac_line" | awk '{print $3}')
            ip_mac_list["$mac"]="$ip"
          fi
      fi
  done < <(sudo nmap -sn --send-eth $network_range)

  # Declare an associative array to store matching rae-ID -> IP
  declare -gA rae_ip_list

  # Loop through the devices list and check if the MAC has a matching IP in ip_mac_list
  for hostname in "${!devices[@]}"; do
      mac_address="${devices[$hostname]}"
      if [[ -n "${ip_mac_list[$mac_address]}" ]]; then
        rae_ip_list["$hostname"]="${ip_mac_list[$mac_address]}"
        echo "found $hostname as ${ip_mac_list[$mac_address]}"
      fi
  done
}