from django.conf import settings
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static

from .views import MainView


urlpatterns = [
    path('', MainView.as_view(), name='main'),
    path('admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
