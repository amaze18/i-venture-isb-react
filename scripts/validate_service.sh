#!/bin/bash

# Check if Streamlit application is running
response=$(curl -s -o /dev/null -w "%{http_code}" http://100.25.194.156:8080/)
if [[ "$response" == "200" ]]; then
    echo "Streamlit application is running."
else
    echo "Streamlit application is not running."
    exit 1
fi
