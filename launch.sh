#!/bin/bash

echo "Starting backend setup..."
 
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi
 
echo "Activating virtual environment..."
# source venv/bin/activate # For Linux/Mac 
source venv/Scripts/activate # For Windows

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading NLTK stopwords..."
python3 -c "import nltk; nltk.download('stopwords')"

echo "Launching FastAPI server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
