#!/bin/bash

echo "Updating pip..."
pip install --upgrade pip

# List of packages to install
PACKAGES=(
    "numpy"
    "matplotlib"
    "pybullet"
)

echo "Installing required packages..."

for package in "${PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install "$package"
done

echo "Installation of required packages complete."
