# encoding: utf-8
# reference:  https://github.com/L1aoXingyu/deploy-pytorch-model

import requests
import argparse

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'


def predict_result(image_path, url=None):
    url = url if url is not None else PyTorch_REST_API_URL
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    r = requests.post(url, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        print('Request success')
        # Loop over the predictions and display them.
        for (i, result) in enumerate(r['predictions']):
            print('{}. {}: {:.4f}'.format(i + 1, result['label'],
                                          result['probability']))
    # Otherwise, the request failed.
    else:
        print('Request failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, help='test image file')
    parser.add_argument('--non_local', action='store_true', default=False, help="non  local access")
    parser.add_argument('--url', type=str, help="")

    args = parser.parse_args()
    if args.non_local:
        url = args.url + '/predict'
    else:
        url = None 
    predict_result(args.file, url)