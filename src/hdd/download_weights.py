import os
from importlib.resources import files

ID_DESC_WEIGHTS = '1He7ELAxJM-RuKS9m4fOfvrtMmRuF-T4P'
ID_DETECTION_WEIGHTS = '1Mrgdh6jd4aWwBwd7fjCQmY1qhOKk1Qad'
ID_BINMASK = '1OFBtTf4SGWRGPSw0MOOyaF584q4iX2jC'


def download_gdrive_files():
    import gdown

    if not os.path.exists(files('hdd.assets.weights') / 'weights_detection.pt'):
        gdown.download(f'https://drive.google.com/uc?/export=download&id={ID_DETECTION_WEIGHTS}',
                       output = files('hdd.assets.weights') / 'weights_detection.pt')

    if not os.path.exists(files('hdd.assets') / 'masks.csv'):
        gdown.download(f'https://drive.google.com/uc?/export=download&id={ID_BINMASK}',
                       output=files('hdd.assets') / 'masks.csv')
        
def main():
    download_gdrive_files()