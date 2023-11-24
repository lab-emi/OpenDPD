#!/bin/bash

get_pt_file() {
    local dir=$1
    local file_count=0
    local pt_file=""

    for file in "$dir"/*; do
         if [[ $file == *.pt ]]; then
            ((file_count++))
            if ((file_count > 1)); then
                echo "Error: Multiple files found with .pt extension."
                exit 1
            fi
            pt_file="$file"
        fi
    done

    if ((file_count == 0)); then
        echo "Error: No file found with .pt extension."
        exit 1
    fi

    echo "$pt_file"
}

