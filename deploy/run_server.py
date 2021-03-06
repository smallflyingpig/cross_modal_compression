# encoding: utf-8
# reference: https://github.com/L1aoXingyu/deploy-pytorch-model
from ImageText.train import load_raw_checkpoint
import io
import json, os
import argparse
import flask
from flask import request, jsonify
from werkzeug.utils import redirect
from base64 import encodebytes
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50
from ImageText.eval import Evaluator as TI_Evaluator
from TextImage.train import Evaluator as IT_Evaluator
from utils.config import cfg_from_file, cfg

from flask import render_template

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
g_it_evaluator = None
g_ti_evaluator = None
use_gpu = torch.cuda.is_available()

"""Utilities
"""
import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

# with open('imagenet_class.txt', 'r') as f:
#    idx2label = eval(f.read())


def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model
    model = resnet50(pretrained=True)
    model.eval()
    if use_gpu:
        model.cuda()


def prepare_image(image, target_size):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """

    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return image

#@app.route("/index")
#def index():
#    return render_template('index.html', username='username', result='')

@app.route("/index", methods=["POST", 'GET'])
def index():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if False and flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # Preprocess the image and prepare it for classification.
            image = prepare_image(image, target_size=(224, 224))

            # Classify the input image and then initialize the list of predictions to return to the client.
            preds = F.softmax(model(image), dim=1)
            results = torch.topk(preds.cpu().data, k=3, dim=1)

            data['predictions'] = list()

            # Loop over the results and add them to the list of returned predictions
            for prob, label in zip(results[0][0], results[1][0]):
                label_name = idx2label[label.item()]
                r = {"label": label_name, "probability": float(prob)}
                data['predictions'].append(r)

            # Indicate that the request was a success.
            data["success"] = True
        rtn_html = 'index.html'
        if flask.request.form.get('project') == 'cross_modal_compression':
            return redirect('cross_modal_compression_mini')
            # rtn_html = 'cross_modal_compression.html'
        elif flask.request.form.get('project') == 'video_super_resolution':
            return redirect('cross_modal_compression')
            # rtn_html = 'cross_modal_compression.html'
        elif flask.request.form.get('project') == 'conceptual_compression':
            return redirect('cross_modal_compression')
            # rtn_html = 'cross_modal_compression.html'
        else:
            rtn_html = 'index.html'
        print(flask.request.form.get('project'), rtn_html)
        username = 'lijiguo_1'
        result = 'sucess'
    else:
        rtn_html = 'index.html'
        username = 'lijiguo'
        result = ''
    # Return the data dictionary as a JSON response.
    # return flask.jsonify(data)
    return render_template(rtn_html, username=username, result=result)


def return_img_stream(img_local_path):
    """
    ????????????:
    ?????????????????????
    :param img_local_path:???????????????????????????????????????
    :return: ?????????
    """
    import base64
    img_stream = ''
    img_local_path = os.path.abspath(img_local_path)
    if os.path.exists(img_local_path):
        with open(img_local_path, 'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
    return img_stream


@app.route("/cross_modal_compression_mini", methods=["POST", 'GET'])
def cross_modal_compression():
    return render_template("cross_modal_compression_mini.html")

def image2text(img):
    des = g_it_evaluator.forward_one_img(img)
    return des

def text2image(text):
    img = g_ti_evaluator.forward_one_sent(text)
    return img 

def Image2Base64(img:Image):
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    # print(encoded_img[:100])
    return u"data:image/png;base64,"+encoded_img


@app.route('/cross_modal_compression_mini/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        # print(request.json)
        json_data = request.json

        img = base64_to_pil(json_data['img'])
        print(f"dataset: {json_data['dataset']}")

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        # preds = model_predict(img, model)

        # Process your result for human
        # pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_proba = '1.0'
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        result = image2text(img) # str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()

        img_rec = Image2Base64(Image.fromarray(text2image(result).transpose(1,2,0)))
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba, rec=img_rec)

    return None

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    parser = argparse.ArgumentParser("flask server")
    parser.add_argument("--non_local", action='store_true', default=False, help="")
    parser.add_argument("--cfg", type=str, default='./cfg/bird_eval.cfg')
    parser.add_argument("--beam_size", type=int, default=4, help="")
    parser.add_argument("--dataset", choices=['coco', 'bird'], default='bird', help="")
    parser.add_argument("--data_dir", type=str, default="./data/birds", help="")
    parser.add_argument("--gpu", type=int, default=0, help="")
    args = parser.parse_args()
    cfg_from_file(args.cfg)
    imsize = cfg.TREE.BASE_SIZE * (2 **(cfg.TREE.BRANCH_NUM - 1))
    g_it_evaluator = TI_Evaluator(cfg, args.dataset, args.data_dir, imsize, args.beam_size, args.gpu)
    g_ti_evaluator = IT_Evaluator(cfg, args.dataset, args.gpu)
    if args.non_local:
        app.run(host='0.0.0.0', port=8090, debug=True)
    else:
        app.run()
    
