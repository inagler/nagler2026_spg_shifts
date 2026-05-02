#!/bin/bash

# Email details
TO="ina.nagler@uib.no"
SUBJECT="change point execution successful"
BODY="All change point runs successfully completed."

# Commands to run with arguments
COMMANDS=(
    "python3 change_point_analysis.py 40 20 4.0 5 true"
    "python3 change_point_analysis.py 40 20 3.0 5 true"
    "python3 change_point_analysis.py 40 20 2.0 5 true"
    "python3 change_point_analysis.py 40 20 1.0 5 true"
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
    BODY="Some change point runs failed:\n$ERRORS"
    SUBJECT="change point execution completed with errors"
fi

# Notify the completion of the script
echo "Batch process completed."

# Send email
echo -e "$BODY" | mail -s "$SUBJECT" $TO

# Print completion message
echo "Email notification sent to $TO."