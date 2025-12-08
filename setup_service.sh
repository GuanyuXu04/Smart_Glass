#!/bin/bash
# Set up systemd service for Smart Glass pipeline on NVIDIA Orin Nano.

chmod +x /home/team15/Documents/SmartGlass/run_pipeline.sh

# Install systemd service
echo "Installing smartglass.service..."
sudo cp /home/team15/Documents/SmartGlass/smartglass.service /etc/systemd/system/

# Reload and enable
echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable smartglass.service

# Start immediately
echo "Starting service..."
sudo systemctl start smartglass.service

echo "Status:"
sudo systemctl status smartglass.service
