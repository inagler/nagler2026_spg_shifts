#!/bin/bash

# Email details
TO="ina.nagler@uib.no"
SUBJECT="averages execution successful"
BODY="Computation of averages has successfully completed."

# Commands to run with arguments
COMMANDS=(
    "/shared_netapp3_home/home/innag3580/.conda/envs/movie/bin/python averages.py PO4"
    #"python3 averages.py TEMP"
    #"python3 averages.py SALT"
    #"python3 averages.py VVEL"
    #"python3 averages.py SHF"
    #"python3 averages.py HMXL"
    #"python3 averages.py TAUX"
    #"python3 averages.py TAUY"
    #"python3 averages.py PSL"
    #"python3 averages.py AICE"
    #"python3 averages.py N_HEAT"
    #"python3 averages.py N_SALT"
    #"python3 averages.py VT"
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
    BODY="Somes computations of averages have failed:\n$ERRORS"
    SUBJECT="averages execution completed with errors"
fi

# Notify the completion of the script
echo "Batch process completed."

# Send email
echo -e "$BODY" | mail -s "$SUBJECT" $TO

# Print completion message
echo "Email notification sent to $TO."