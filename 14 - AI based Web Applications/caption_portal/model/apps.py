import os

from django.apps import AppConfig
from django.conf import settings

from .generate_caption import load_model, load_word_map


class ModelConfig(AppConfig):
    name = 'model'
    model_path = os.path.join(settings.STATIC_ROOT, 'models', 'model.pth.tar')
    wordmap_path = os.path.join(settings.STATIC_ROOT, 'models', 'wordmap.json')

    word_map, rev_word_map = load_word_map(wordmap_path)
    encoder, decoder = load_model(model_path)
