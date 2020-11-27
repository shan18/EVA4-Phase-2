try:
    import unzip_requirements
except ImportError:
    pass

import os
import json
import base64
from requests_toolbelt.multipart import decoder


def fetch_input(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final input that will be used by the model
    audio = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Audio obtained')
    
    return audio.content


def convert(event, context):
    try:
        # Get audio from the request
        audio_file = '/tmp/temp.wav'
        audio_file_processed = '/tmp/temp_processed.wav'
        audio = fetch_input(event)
        print('Type:', type(audio))

        # Do some pre-processing
        with open(audio_file, 'wb') as f:
            f.write(audio)
        print('File written')

        # Convert
        print('Converting')
        os.system(f'./ffmpeg -i {audio_file} {audio_file_processed}')
        with open(audio_file_processed, 'rb') as f:
            output = f.read()
        output_bytes = base64.b64encode(output)
        print('Converted')

        # Remove file
        os.remove(audio_file)
        os.remove(audio_file_processed)
        print('files removed')

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'data': output_bytes.decode('ascii')})
        }
    except Exception as e:
        print(repr(e))
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'error': repr(e)})
        }
