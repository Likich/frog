import re
import joblib
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import argparse
from sklearn.pipeline import Pipeline
import pickle

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[_0-9]', ' ', text)
    text = re.sub(r'\s\s+', ' ', text)
    return text

# Load the Naive Bayes model, TF-IDF vectorizer, and label encoder


tfidf = joblib.load('tfidf_vectorizer.pkl')
enc = joblib.load('label_encoder.pkl')
nb = joblib.load('multinomial_nb_model.pkl')


# # Load your label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = ['Achinese', 'Afrikaans', 'Akan', 'Albanian', 'Amharic', 'Arabic',
       'Armenian', 'Assamese', 'Asturian', 'Awadhi', 'Aymara',
       'Azerbaijani', 'Balinese', 'Bambara', 'Bashkir', 'Basque',
       'Belarusian', 'Bemba (Zambia)', 'Bengali', 'Bhojpuri', 'Buginese',
       'Bulgarian', 'Burmese', 'Catalan', 'Cebuano',
       'Central Atlas Tamazight', 'Chhattisgarhi', 'Chinese', 'Chokwe',
       'Crimean Tatar', 'Czech', 'Danish', 'Dinka', 'Dutch', 'Dyula',
       'Dzongkha', 'English', 'Esperanto', 'Estonian', 'Ewe', 'Faroese',
       'Fijian', 'Finnish', 'Fon', 'French', 'Friulian', 'Fulah',
       'Galician', 'Ganda', 'Georgian', 'German', 'Guarani', 'Gujarati',
       'Haitian', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic',
       'Igbo', 'Iloko', 'Irish', 'Italian', 'Japanese', 'Javanese',
       'Kabiyè', 'Kabuverdianu', 'Kabyle', 'Kachin', 'Kamba (Kenya)',
       'Kannada', 'Kanuri', 'Kashmiri', 'Kazakh', 'Khmer', 'Kikuyu',
       'Kimbundu', 'Kinyarwanda', 'Kirghiz', 'Kongo', 'Korean', 'Kurdish',
       'Lao', 'Latvian', 'Ligurian', 'Limburgan', 'Lingala', 'Lithuanian',
       'Lombard', 'Luba-Lulua', 'Luo (Kenya and Tanzania)', 'Lushai',
       'Luxembourgish', 'Macedonian', 'Magahi', 'Maithili', 'Malagasy',
       'Malay (macrolanguage)', 'Malayalam', 'Maltese', 'Manipuri',
       'Maori', 'Marathi', 'Modern Greek (1453-)', 'Mongolian', 'Mossi',
       'Nepali (macrolanguage)', 'Norwegian', 'Nuer', 'Nyanja',
       'Occitan (post 1500)', 'Oriya (macrolanguage)', 'Oromo',
       'Pangasinan', 'Panjabi', 'Papiamento', 'Pedi', 'Persian', 'Polish',
       'Portuguese', 'Pushto', 'Quechua', 'Romanian', 'Rundi', 'Russian',
       'Samoan', 'Sango', 'Sanskrit', 'Santali', 'Sardinian',
       'Scottish Gaelic', 'Serbo-Croatian', 'Shan', 'Shona', 'Sicilian',
       'Silesian', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali',
       'Southern Sotho', 'Spanish', 'Sundanese',
       'Swahili (macrolanguage)', 'Swati', 'Swedish', 'Tagalog', 'Tajik',
       'Tamashek', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan',
       'Tigrinya', 'Tok Pisin', 'Tsonga', 'Tswana', 'Tumbuka', 'Turkish',
       'Turkmen', 'Uighur', 'Ukrainian', 'Umbundu', 'Urdu', 'Uzbek',
       'Venetian', 'Vietnamese', 'Waray (Philippines)', 'Welsh', 'Wolof',
       'Xhosa', 'Yiddish', 'Yoruba', 'Zulu']  # Replace with your actual language classes

# use pipeline to combine prefitted vectorizer and trained model into one object
model = Pipeline([('vectorizer',tfidf),('nb',nb)])
# function to predict language from text
def predict_language(text):
    pred = model.predict([clean_text(text)])
    ans = enc.inverse_transform(pred)
    return ans[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect the language of a given text.")
    parser.add_argument("--text", type=str, help="Input text to detect language.")
    
    args = parser.parse_args()
    
    if args.text:
        detected_language = predict_language(args.text)
        print(f"Predicted Language: {detected_language}")
    else:
        print("Please provide text using the --text argument.")