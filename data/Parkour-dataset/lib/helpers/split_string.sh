#!/bin/bash

# Split a string with a user-specified delimiter
# Usage:
# > split_string alice,bob,eve ,
split_string() {
    string=${1}
    delimiter=${2}

    # Convert string to array
    IFS=${delimiter} read -ra NAMES <<< "$string"

    # Print all names from array
    for i in "${NAMES[@]}"; do
        echo $i
    done
}
