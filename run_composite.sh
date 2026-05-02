#!/bin/bash

# Email details
TO="ina.nagler@uib.no"
SUBJECT="composite execution successful"
BODY="Composite runs successfully completed."

# Commands to run with arguments
COMMANDS=(
    "python3 composite_analysis.py N_HEAT change_point_indices_2.0_40_20_20.csv aa_hist"
    "python3 composite_analysis.py N_HEAT change_point_indices_3.0_40_20_20.csv ghg_hist"
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
    BODY="Some composite runs failed:\n$ERRORS"
    SUBJECT="composite execution completed with errors"
fi


# Send email
echo -e "$BODY" | mail -s "$SUBJECT" $TO

# Print completion message
echo "Email notification sent to $TO."