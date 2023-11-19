#!/bin/bash

# Split a string with a delimiter
# example: split_string barbell,hammer,scythe,spade ,
split_string() {
    string=${1}
    delimiter=${2}

    #string="barbell,hammer,scythe,spade"  #String with names
    IFS=${delimiter} read -ra NAMES <<< "$string"    #Convert string to array

    #Print all names from array
    for i in "${NAMES[@]}"; do
        echo $i
    done
}
