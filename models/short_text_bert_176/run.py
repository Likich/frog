import torch
import argparse
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import gdown
import subprocess

# model_drive_link = 'https://drive.google.com/file/d/1-8d412OfxwYW5gjw4TsiiyONGez0HhAV/'

# # Define the local path to save the downloaded model file
# local_model_path = '/finetuned_bert.pth'
# # Download the model from Google Drive if it doesn't exist locally

# wget_command = f'wget --no-check-certificate "{model_drive_link}" -O "{local_model_path}"'
# subprocess.call(wget_command, shell=True)
# Load your pretrained BERT model

model = torch.load('short_text_bert.pth')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load your label encoder
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

# Function to predict the language of a sentence
def predict_language(sentence, model, device):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    model.eval()
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_language = label_encoder.classes_[predicted_class]
    return predicted_language


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect the language of a given text.")
    parser.add_argument("--text", type=str, help="Input text to detect language.")
    
    args = parser.parse_args()
    
    if args.text:
        detected_language = predict_language(args.text, model, 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Predicted Language: {detected_language}")
    else:
        print("Please provide text using the --text argument.")
