#!/bin/bash
set -e

# remove Volumes 
echo "remove volumes for mapping to containers in docker-compose.yml"
# 1. - $PWD/app_data:/app/data
sudo rm -rf ../app_data

# 2. -  $PWD/app_data_backups:/app/data_backups
sudo rm -rf ../app_data_backups

# 3. - $PWD/data:/data
sudo rm -rf ../data

# 4. - $PWD/backups:/backups
sudo rm -rf ../backups

# 5. - $PWD/chroma-data:/data
sudo rm -rf ../chroma-data

# 6. - $PWD/backups_chroma:/backups
sudo rm -rf ../backups_chroma

# sudo rm -rf ~/.docker
# sudo systemctl restart docker 