# FROG: Fantastic Recognition Of lanGuages
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub license](https://img.shields.io/github/license/SpirinEgor/gulag)](https://github.com/Likich/frog/blob/master/LICENSE)

![frog](https://github.com/Likich/frog/assets/52376183/a92d606b-a2ba-4839-bc2e-96c0cb27cbb0)

This repository contains a comprehensive language classification solution using an ensemble model that classifies languages based on macro-families and then further refines the classification to identify specific languages within each family. The classification model is trained on a diverse dataset comprising data from Wikipedia, Flores, and Tatoeba, making it a robust and accurate language identification system.

```
> Привет! Меня зовут Лика.
Predicted language: Russian
> Hello! My name is Lika.
Predicted language: English
```

For this specific task, only 306 languages can be detected because of computational limits (training on Colab). However, the combined 455 languages dataset is available and can be readily used for this model as well. The model still can distinguish Tatar and Crimean Tatar!

# Datasets

To make the analysis easier, some languages were combined into macro-groups. For example, 9 types of Arabic were hardly distinguishable by the model, so they were combined into Arabic (macro).

Flores dataset with macro languages is available following this [link](https://drive.google.com/file/d/1rn_OMO0HGejUVoYUrZ96s7XfiY981gBA/view?usp=sharing). 
A custom dataset with 306 languages is available following this [link](https://drive.google.com/file/d/1WTaLUB5oo26QyU_zQTWGhmmfCI4f3LcD/view?usp=sharing). 
The largest dataset with 455 languages is available [here](https://drive.google.com/file/d/1cTIOoM1bhZod1TNXqZPBK24WPgQvfSoj/view?usp=sharing). However it is too large for inference right now.

# Training

Training was performed in google colab using mainly 2 notebooks, that you can find in a respective folder. [First notebook](https://github.com/Likich/frog/blob/main/notebooks/Language_Identification_Training_initial.ipynb) contains training of Multinomial Naive Bayes, Bi-LSTM, BERT with freezing weights and finetunined BERT on Flores dataset using only macro languages (176). [Second notebook](https://github.com/Likich/frog/blob/main/notebooks/Combined_finetuning_ipynb_.ipynb) continues the transfer learning of finetuned BERT model extending classification to 306 languages, however with lower accuracy, but still high for earlier predicted languages. 

# Inference

The main script is containesd in main.py.
To use main.py, you can run it from the command line and provide the text as an argument like this:

```
python3 main.py --text "Your input text goes here."
```
The script will use the downloaded pretrained model to predict the language of the input text and print the predicted language class.

The final model is stored in [model link](https://drive.google.com/file/d/1-8d412OfxwYW5gjw4TsiiyONGez0HhAV/view?usp=drive_link). Please download it first locally for further usage. This model was trained initially on Flores dataset with 176 languages for 4 hours on Tesla T4 GPU. Then finetuned on larger dataset containing 306 languages for 7 hours.

To train the model you can use respective notebooks and run the on Colab GPU.

# Models

4 models are presented in this repository. Please download models to the general folder prior to running the code. Here you can find respective links to download necessary models:

[Finetuned_bert_176](https://drive.google.com/file/d/1wBnF6-2CvZWcPCqmVdEdgG7V8MNFsRYa/view?usp=sharing) multilingual bert that was initially trained on 104 languages and now finetuned on 176 dataset. Finetuning for 2 epochs (2 hours). 99% accuracy.
[Finetuned_bert_306](https://drive.google.com/file/d/1-8d412OfxwYW5gjw4TsiiyONGez0HhAV/view?usp=drive_link) previous model finetuned on 306 languages for 5 epochs (7 hours). 86% accuracy.
[Naive_Bayes_176](https://drive.google.com/drive/folders/1etNjij2Uf-01xYcuYtrHQBcLqa0Z4Z8a?usp=sharing) predicting 176 languages. 97% accuracy. Missclassifies Japanese completely.
[Short_text_bert_176](https://drive.google.com/file/d/1lflgIGFuY7J6o5KQN13dYpZmIXDp0bFa/view?usp=sharing) finetuned on texts shorter than 20 characters, achieves 83% accuracy.

Using run.py you can run inference on respective models in each folder. Usage:

```
python3 models/short_text_bert_176/run.py --text "Your input text goes here."
```