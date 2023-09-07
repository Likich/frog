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

model = torch.load('finetuned_306_v2_bert.pth')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load your label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = ['Achinese', 'Afrikaans', 'Ainu (Japan)', 'Akan', 'Albanian',
       'Algerian Arabic', 'Amharic', 'Ancient Greek (to 1453)',
       'Ancient Hebrew', 'Arabic', 'Aragonese', 'Armenian', 'Assamese',
       'Asturian', 'Awadhi', 'Ayacucho Quechua', 'Azerbaijani',
       'Balinese', 'Baluchi', 'Bambara', 'Banjar', 'Bashkir', 'Basque',
       'Bavarian', 'Belarusian', 'Bemba (Zambia)', 'Bengali', 'Bhojpuri',
       'Bosnian', 'Breton', 'Brithenig', 'Buginese', 'Bulgarian',
       'Buriat', 'Burmese', 'Catalan', 'Cebuano',
       'Central Atlas Tamazight', 'Central Aymara', 'Central Kanuri',
       'Central Kurdish', 'Chamorro', 'Chavacano', 'Chhattisgarhi',
       'Chinese', 'Choctaw', 'Chokwe', 'Chukot', 'Chuvash',
       'Congo Swahili', 'Cornish', 'Creek', 'Crimean Tatar', 'Croatian',
       'Czech', 'Danish', 'Dari', 'Dutch', 'Dyula', 'Dzongkha',
       'Eastern Mari', 'Eastern Yiddish', 'Egyptian Arabic', 'Emilian',
       'English', 'Esperanto', 'Estonian', 'Evenki', 'Ewe', 'Faroese',
       'Fijian', 'Finnish', 'Fon', 'French', 'Friulian', 'Galician',
       'Ganda', 'Georgian', 'German', 'Gothic', 'Gronings',
       'Guadeloupean Creole French', 'Guarani', 'Gujarati', 'Gulf Arabic',
       'Haitian', 'Halh Mongolian', 'Hausa', 'Hawaiian', 'Hebrew',
       'Hindi', 'Ho', 'Hungarian', 'Hunsrik', 'Icelandic', 'Ido', 'Igbo',
       'Iloko', 'Indonesian',
       'Interlingua (International Auxiliary Language Association)',
       'Interlingue', 'Iranian Persian', 'Irish', 'Italian', 'Japanese',
       'Javanese', 'Jewish Babylonian Aramaic (ca. 200-1200 CE)',
       'Kabiyè', 'Kabuverdianu', 'Kabyle', 'Kachin', 'Kadazan Dusun',
       'Kalaallisut', 'Kalmyk', 'Kamba (Kenya)', 'Kannada',
       'Karachay-Balkar', 'Karelian', 'Kashmiri', 'Kashubian', 'Kazakh',
       'Khasi', 'Khmer', 'Kikuyu', 'Kimbundu', 'Kinyarwanda', 'Kirghiz',
       'Klingon', 'Kongo', 'Korean', 'Kotava', 'Kumyk', 'Kven Finnish',
       'Ladino', 'Lao', 'Latgalian', 'Latin', 'Laz', 'Levantine Arabic',
       'Ligurian', 'Limburgan', 'Lingala', 'Lingua Franca Nova',
       'Literary Chinese', 'Lithuanian', 'Lojban', 'Lombard',
       'Low German', 'Lower Sorbian', 'Luba-Lulua',
       'Luo (Kenya and Tanzania)', 'Lushai', 'Luxembourgish', 'Láadan',
       'Macedonian', 'Magahi', 'Maithili', 'Malay (individual language)',
       'Malayalam', 'Maltese', 'Mandarin Chinese', 'Manipuri', 'Maori',
       'Mapudungun', 'Marathi', 'Mesopotamian Arabic',
       'Middle English (1100-1500)', 'Mikasuki', 'Minangkabau',
       'Mingrelian', 'Modern Greek (1453-)', 'Mongolian',
       'Moroccan Arabic', 'Mossi', 'Najdi Arabic', 'Nande',
       'Nepali (individual language)', 'Nigerian Fulfulde', 'Nogai',
       'North Azerbaijani', 'North Moluccan Malay', 'Northern Frisian',
       'Northern Kurdish', 'Northern Sami', 'Northern Uzbek',
       'Norwegian Bokmål', 'Norwegian Nynorsk', 'Novial', 'Nuer',
       'Nyanja', 'Occitan (post 1500)', 'Odia',
       'Old English (ca. 450-1100)', 'Old Russian',
       'Oriya (macrolanguage)', 'Ossetian', 'Ottoman Turkish (1500-1928)',
       'Pampanga', 'Pangasinan', 'Panjabi', 'Papiamento', 'Pattani Malay',
       'Pedi', 'Persian', 'Picard', 'Piemontese', 'Plateau Malagasy',
       'Polish', 'Portuguese', 'Prussian', 'Pushto', 'Quechua', 'Quenya',
       'Rohingya', 'Romanian', 'Romany', 'Rundi', 'Russian', 'Rusyn',
       'Samoan', 'Sango', 'Sanskrit', 'Santali', 'Saraiki', 'Sardinian',
       'Scottish Gaelic', 'Serbian', 'Shan', 'Shona', 'Sicilian',
       'Silesian', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian', 'Somali',
       'South Azerbaijani', 'Southern Kurdish', 'Southern Pashto',
       'Southern Sotho', 'Southwestern Dinka', 'Spanish', 'Sranan Tongo',
       'Standard Arabic', 'Standard Latvian', 'Standard Malay',
       'Standard Moroccan Tamazight', 'Sumerian', 'Sundanese', 'Swabian',
       'Swahili (individual language)', 'Swati', 'Swedish',
       'Swiss German', "Ta'izzi-Adeni Arabic", 'Tachawit', 'Tachelhit',
       'Tagalog', 'Tahaggart Tamahaq', 'Tajik', 'Talossan', 'Tamasheq',
       'Tamil', 'Tarifit', 'Tase Naga', 'Tatar', 'Telugu', 'Tetum',
       'Thai', 'Tibetan', 'Tigre', 'Tigrinya', 'Tok Pisin', 'Toki Pona',
       'Tosk Albanian', 'Tsonga', 'Tswana', 'Tumbuka', 'Tunisian Arabic',
       'Turkish', 'Turkmen', 'Twi', 'Udmurt', 'Uighur', 'Ukrainian',
       'Umbundu', 'Upper Sorbian', 'Urdu', 'Uzbek', 'Venetian',
       'Vietnamese', 'Volapük', 'Waray (Philippines)', 'Wayuu', 'Welsh',
       'West Central Oromo', 'Western Frisian', 'Wolof', 'Wu Chinese',
       'Xhosa', 'Yakut', 'Yiddish', 'Yoruba', 'Yue Chinese', 'Zaza',
       'Zulu']  # Replace with your actual language classes

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
