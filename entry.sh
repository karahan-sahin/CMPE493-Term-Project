echo "Starting environment setting..."
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
python3 -c "import nltk; nltk.download('all')"
echo "Environment setting has been finished."
echo "You are ready to start the app."
deactivate
