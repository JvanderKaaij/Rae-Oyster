#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_filepath>"
    exit 1
fi

# Store the arguments in variables
input_file=$(realpath "$1")
output_filepath="$2"

if [ ! -f "$input_file" ]; then
    echo "Error: $input_file does not exist."
    exit 1
fi

# Source the script that contains the find_rae_ip_list function
source ./find_rae_ip_list.sh

# Call the function to populate the rae_ip_list
find_rae_ip_list

rae_count=${#rae_ip_list[@]}
echo "Total number of RAE's found: $rae_count"

# Now rae_ip_list is populated, you can access it here
echo "Final list of rae-ID -> IP:"
for rae_id in "${!rae_ip_list[@]}"; do
    echo "rae-ID: $rae_id | IP: ${rae_ip_list[$rae_id]}"
    echo "Attempting to copy $input_file to $output_filepath"
    scp "$input_file" root@${rae_ip_list[$rae_id]}:~/$(basename "$output_filepath")

    if [ $? -ne 0 ]; then
        echo "Failed to copy $input_file to ${rae_ip_list[$rae_id]}:$output_filepath"
    else
        echo "Successfully copied $input_file to ${rae_ip_list[$rae_id]}:$output_filepath"
    fi
done