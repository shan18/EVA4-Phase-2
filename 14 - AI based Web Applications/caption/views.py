import os
import random

from django.conf import settings
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import View
from django.http import HttpResponseRedirect, JsonResponse

from PIL import Image

from model.apps import ModelConfig
from model.generate_caption import caption_image


class MainView(View):

    def __init__(self, *args, **kwargs):
        self.image_name = 'uploaded_image.jpg'
        super().__init__(*args, **kwargs)

    def process_image(self, image):
        image = Image.open(image)
        image = image.convert('RGB')
        image.save(os.path.join(settings.MEDIA_ROOT, self.image_name))
        return image

    def get(self, request, *args, **kwargs):
        return render(request, 'main.html', {})
    
    def post(self, request, *args, **kwargs):
        image = self.process_image(request.FILES['image'])
        caption = caption_image(
            image,
            ModelConfig.encoder,
            ModelConfig.decoder,
            ModelConfig.word_map,
            ModelConfig.rev_word_map
        )
        if request.is_ajax():
            return JsonResponse({
                'success': True,
                'image': f'{settings.MEDIA_URL}{self.image_name}',
                'caption': caption
            })
        url = reverse('main')
        return HttpResponseRedirect()
