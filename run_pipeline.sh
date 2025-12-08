#!/bin/bash
# Auto-connect to ESP32 WiFi and run Smart Glass pipeline on NVIDIA Orin Nano.

# Connect to fanghb (ESP32-hosted WiFi network)
nmcli connection up fanghb

# Run pipeline as team15 user with full environment
su - team15 -c "cd /home/team15/Documents/SmartGlass && /usr/bin/python3 pipeline.py --esp-host 192.168.4.1 --loop --nav-host 127.0.0.1 --log INFO > /home/team15/pipeline.log 2>&1"
