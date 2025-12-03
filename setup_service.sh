#!/bin/bash

# Make the runner script executable
chmod +x /home/team15/Documents/SmartGlass/run_pipeline.sh

# Copy the service file to systemd directory
echo "Installing smartglass.service to /etc/systemd/system/..."
sudo cp /home/team15/Documents/SmartGlass/smartglass.service /etc/systemd/system/

# Reload systemd daemon
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the service to start on boot
echo "Enabling smartglass.service..."
sudo systemctl enable smartglass.service

# Start the service immediately (optional)
echo "Starting smartglass.service..."
sudo systemctl start smartglass.service

echo "Status of smartglass.service:"
sudo systemctl status smartglass.service
