#!/bin/bash

# Email details
TO="ina.nagler@uib.no"
SUBJECT="stds execution successful"
BODY="Computation of standard deviations has successfully completed."

# Commands to run with arguments
COMMANDS=(
    "python3 standard_deviations.py TEMP"
    "python3 standard_deviations.py SALT"
    "python3 standard_deviations.py VVEL"
    "python3 standard_deviations.py SHF"
    "python3 standard_deviations.py HMXL"
    "python3 standard_deviations.py TAUX"
    "python3 standard_deviations.py TAUY"
    "python3 standard_deviations.py PSL"
    "python3 standard_deviations.py AICE"
    "python3 standard_deviations.py N_HEAT"
    "python3 standard_deviations.py N_SALT"
    "python3 standard_deviations.py VT"
)

# Initialize an empty string to collect errors
ERRORS=""

# Notify the start of the script
echo "Starting batch process..."

# Run the commands
for COMMAND in "${COMMANDS[@]}"; do
    $COMMAND
    if [ $? -ne 0 ]; then
        ERRORS+="The code execution failed on command: $COMMAND\n"
        echo "Error: The code execution failed on command: $COMMAND"
    fi
done

# Check if there were any errors
if [ -n "$ERRORS" ]; then
    BODY="Somes computations of standard devaitions have failed:\n$ERRORS"
    SUBJECT="stds execution completed with errors"
fi

# Notify the completion of the script
echo "Batch process completed."

# Send email
echo -e "$BODY" | mail -s "$SUBJECT" $TO

# Print completion message
echo "Email notification sent to $TO."