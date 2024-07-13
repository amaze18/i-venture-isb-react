#!/bin/bash

# Start Streamlit application
nohup streamlit run /opt/codedeploy-agent/deployment-root/streamlit_app.py &
echo -ne '\n'
