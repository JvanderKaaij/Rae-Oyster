#!/bin/bash

# Source the script that contains the find_rae_ip_list function
source ./find_rae_ip_list.sh

# Call the function to populate the rae_ip_list
find_rae_ip_list

rae_count=${#rae_ip_list[@]}
echo "Total number of RAE's found: $rae_count"

# Now rae_ip_list is populated, you can access it here
echo "Final list of rae-ID -> IP:"
for rae_id in "${!rae_ip_list[@]}"; do
    echo "rae-ID: $rae_id | IP: ${rae_ip_list[$rae_id]} Shutting Down"
    ssh -n -o BatchMode=yes root@${rae_ip_list[$rae_id]} "sudo systemctl enable wpa_supplicant.service" &
done