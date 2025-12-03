#!/bin/bash
# Switch internet connection (running as root)
nmcli connection up fanghb

# Run the pipeline script as user team15
# We use 'su - team15 -c' to ensure we load the user's environment (PATH, python packages, etc.)
su - team15 -c "cd /home/team15/Documents/SmartGlass && /usr/bin/python3 pipeline.py --esp-host 192.168.4.1 --loop --nav-host 127.0.0.1 --log INFO > /home/team15/pipeline.log 2>&1"
