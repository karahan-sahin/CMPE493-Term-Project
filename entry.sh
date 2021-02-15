echo "Starting environment setting..."
python3 -m venv venv
source ./venv/bin/activate
pip3 install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
echo "Environment setting has been finished."
echo "You are ready to start the app."
deactivate
