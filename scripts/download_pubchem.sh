#!/bin/bash

# Set the target directory
TARGET_DIR="/home/dfl32/scratch/pubchem"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Change to the target directory
cd "$TARGET_DIR"

# Function to download and verify a file
download_and_verify() {
    local url="$1"
    local filename=$(basename "$url")
    
    echo "Downloading $filename..."
    wget "$url"
    
    if [ $? -ne 0 ]; then
        echo "Failed to download $filename"
        return 1
    fi
    
    echo "Verifying $filename..."
    wget "${url}.md5"
    md5sum -c "${filename}.md5"
    
    if [ $? -ne 0 ]; then
        echo "MD5 verification failed for $filename"
        return 1
    fi
    
    echo "$filename downloaded and verified successfully"
    return 0
}

# Download Compound data
compound_url="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/XML/Compound_000000001_000500000.xml.gz"
download_and_verify "$compound_url"

# Download Substance data
substance_url="https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/CURRENT-Full/XML/Substance_000000001_000500000.xml.gz"
download_and_verify "$substance_url"

# Download Bioassay data
bioassay_url="https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/XML/1000001_1001000.xml.gz"
download_and_verify "$bioassay_url"

# Uncompress the files
echo "Uncompressing downloaded files..."
gunzip *.gz

echo "Download and verification process completed."