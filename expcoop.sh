#!/bin/bash

# Set the path to your XML file
xml_file="/home/tianyu/code/SUMO-changing-lane-agent/networks/merge_qew_multi_test/qew_mississauga_rd.rou-short.xml"

# Path to Python script
python_script="example/merge_step_test.py"

# Directory to store XML files
xml_dir="/home/tianyu/code/SUMO-changing-lane-agent/results/"

# Array of cooperative values
cooperative_values=("0" "0.2" "0.4" "0.6" "0.8" "1")

# Single log file
log_file="script_log.txt"

# Clear existing log file
> "$log_file"

# Run the script for each cooperative value
for coop_value in "${cooperative_values[@]}"; do
    # Create the XML file path based on the cooperative value
    stats_path="${xml_dir}stat${coop_value}.xml"

    # Create the XML file
    echo "<statistics></statistics>" > "$stats_path"

    # Use sed to replace the lcCooperative value in the SUMO configuration file
    sed -i "s/\(<vType id=\"human\".*lcCooperative=\"\)[^\"]*\(\".*\/>\)/\1$coop_value\2/" "$xml_file"

    echo "Updated lcCooperative value to $coop_value"

    # Update the Python script with the new XML file path
    sed -i "73s|self.stats_path=.*|self.stats_path='$stats_path'|" "/home/tianyu/code/SUMO-changing-lane-agent/gym_sumo/gym_sumo/envs/sumo_env_merge_net.py"

    # Run the Python script, filter out unwanted lines, and append output to the log file
    python "$python_script" --coop "$coop_value" 2>&1 | grep -vE "Error: Answered with error|step [0-9]+" >> "$log_file"

    # Append the cooperative value to the log file
    echo "cooperative_value: $coop_value" >> "$log_file"

    echo "Script execution complete for cooperative value $coop_value"
done

echo "All script executions complete. Log saved to $log_file"
