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

The final model is stored in [model link]([https://wandb.ai/voudy/gulag/runs/a55dbee8?workspace=user-voudy](https://drive.google.com/file/d/1-8d412OfxwYW5gjw4TsiiyONGez0HhAV/view?usp=drive_link)https://drive.google.com/file/d/1-8d412OfxwYW5gjw4TsiiyONGez0HhAV/view?usp=drive_link). Please download it first locally for further usage.
