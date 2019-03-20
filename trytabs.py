import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import os
import base64
import cv2
import numpy as np
from flask import Flask, send_from_directory
import gc
from urllib.parse import quote as urlquote
import imutils
import pytesseract
from PIL import Image
import time
import random
from datetime import datetime
import datetime

UPLOAD_DIRECTORY = "/project/app_uploaded_files"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

image_filename = UPLOAD_DIRECTORY + '/metro2.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=external_stylesheets)


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


app.config['suppress_callback_exceptions'] = True
app.scripts.config.serve_locally = True

app.layout = html.Div([
    html.H1('Dash Tabs component demo'),
    dcc.Upload(
        id="upload-data",
        children=html.Div(
            ["Drag and drop or click to select a file to upload."]
        ),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
        },
        multiple=False,
    ),
    dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Tab One: Opencv', value='tab-1-example'),
        dcc.Tab(label='Tab Two: Mask-RCNN', value='tab-2-example'),
        dcc.Tab(label='Tab Three: YOLO', value='tab-3-example'),
        dcc.Tab(label='Tab Four: Nothing', value='tab-4-example'),
        dcc.Tab(label='Tab Five: HSV example', value='tab-5-example')
    ]),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='tabs-content-example'),
    html.Div(id='all-in-one-img'),
    html.H2("File List"),
    html.Ul(id="file-list"),
])


@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return html.Div([
                html.H3('Tab content 1'),
                html.Div(id='image-view'),
                html.Label('Blur'),
                dcc.RadioItems(
                    id='blur',
                    options=[
                        {'label': 'Gaussian Blurring', 'value': '1'},
                        {'label': 'Median Blur', 'value': '2'},
                        {'label': 'bilateral Blur', 'value': '3'},
                        {'label': 'Averaging', 'value': '4'},
                        {'label': 'None', 'value': '5'}
                    ],
                    value='5'
                ),
                html.Label('Thresh'),
                dcc.Slider(
                    id='thresh',
                    min=1,
                    max=10,
                    step=1,
                    marks=dict(zip([i for i in range(0, 10)], [str(i) for i in range(0, 10)])),
                    value=1,
                    included=False
                ),
                html.Label('Morph'),
                dcc.RadioItems(
                    id='morph',
                    options=[
                        {'label': 'Erosion', 'value': '1'},
                        {'label': 'Dilation', 'value': '2'},
                        {'label': 'Opening', 'value': '3'},
                        {'label': 'Closing', 'value': '4'},
                        {'label': 'None', 'value': '5'}
                    ],
                    value='5'
                ),
                html.Div(id='my-div-1')
        ])
    elif tab == 'tab-2-example':
        return html.Div([
            html.H3('Tab content 2'),
            html.Label('Radio Items'),
            html.Label('Confidence'),
            dcc.Slider(
                id='first-slider',
                min=0,
                max=1,
                step=0.1,
                marks=dict(zip([i/10 for i in range(0, 10)], [str(i/10) for i in range(0, 10)])),
                value=0.1,
                included=False
            ),
            html.Label('Threshold'),
            dcc.Slider(
                id='second-slider',
                min=0,
                max=1,
                step=0.1,
                marks=dict(zip([i/10 for i in range(0, 10)], [str(i/10) for i in range(0, 10)])),
                value=0.1,
                included=False
            ),
            html.Label('Class'),
            dcc.Input(id='class-selector', type='text', value=None, placeholder='Choose a class to track'),
            html.Div(id='my-div-2')
        ])
    elif tab == 'tab-3-example':
        return html.Div([
            html.H3('Tab content 3'),
            html.Label('Radio Items'),
            html.Label('Confidence'),
            dcc.Slider(
                id='first-slider',
                min=0,
                max=1,
                step=0.1,
                marks=dict(zip([i/10 for i in range(0, 10)], [str(i/10) for i in range(0, 10)])),
                value=0.1,
                included=False
            ),
            html.Label('Threshold'),
            dcc.Slider(
                id='second-slider',
                min=0,
                max=1,
                step=0.1,
                marks=dict(zip([i/10 for i in range(0, 10)], [str(i/10) for i in range(0, 10)])),
                value=0.1,
                included=False
            ),
            html.Label('Class'),
            dcc.Input(id='class-selector', type='text', value=None, placeholder='Choose a class to track'),
            html.Div(id='my-div-3')
        ])
    elif tab == 'tab-4-example':
        return html.Div([
            html.H3('Tab content 4'),
            html.Label('Radio Items'),
            html.Label('Blur'),
            dcc.RadioItems(
                id='blur',
                options=[
                    {'label': 'Gaussian Blurring', 'value': '1'},
                    {'label': 'Median Blur', 'value': '2'},
                    {'label': 'bilateral Blur', 'value': '3'},
                    {'label': 'Averaging', 'value': '4'},
                    {'label': 'None', 'value': '5'}
                ],
                value='5'
            ),
            html.Label('Thresh'),
            dcc.Slider(
                id='thresh',
                min=1,
                max=10,
                step=1,
                marks=dict(zip([i for i in range(0, 10)], [str(i) for i in range(0, 10)])),
                value=1,
                included=False
            ),
            html.Label('Morph'),
            dcc.RadioItems(
                id='morph',
                options=[
                    {'label': 'Erosion', 'value': '1'},
                    {'label': 'Dilation', 'value': '2'},
                    {'label': 'Opening', 'value': '3'},
                    {'label': 'Closing', 'value': '4'},
                    {'label': 'None', 'value': '5'}
                ],
                value='5'
            ),
            html.Label('Resize'),
            dcc.Input(id='resize', type='float', value=1.0, placeholder='Choose resize'),
            html.Div(id='output-text'),
            html.Div(id='my-div-4')
        ])
    elif tab == 'tab-5-example':
            return html.Div([
                    html.H3('Tab content 5'),
                    html.Div(id='image-view'),
                    html.Label('Choose HSV'),
                    html.Label('H1'),
                    dcc.Input(id='h1', type='int', value=1, placeholder='Choose H'),
                    html.Label('S1'),
                    dcc.Input(id='s1', type='int', value=1, placeholder='Choose S'),
                    html.Label('V1'),
                    dcc.Input(id='v1', type='int', value=1, placeholder='Choose V'),
                    html.Label('H2'),
                    dcc.Input(id='h2', type='int', value=1, placeholder='Choose H'),
                    html.Label('S2'),
                    dcc.Input(id='s2', type='int', value=1, placeholder='Choose S'),
                    html.Label('V2'),
                    dcc.Input(id='v2', type='int', value=1, placeholder='Choose V'),
                    html.Label('Blur'),
                    dcc.RadioItems(
                        id='blur',
                        options=[
                            {'label': 'Gaussian Blurring', 'value': '1'},
                            {'label': 'Median Blur', 'value': '2'},
                            {'label': 'bilateral Blur', 'value': '3'},
                            {'label': 'Averaging', 'value': '4'},
                            {'label': 'None', 'value': '5'}
                        ],
                        value='5'
                    ),
                    html.Label('Thresh'),
                    dcc.Slider(
                        id='thresh',
                        min=1,
                        max=10,
                        step=1,
                        marks=dict(zip([i for i in range(0, 10)], [str(i) for i in range(0, 10)])),
                        value=1,
                        included=False
                    ),
                    html.Label('Morph'),
                    dcc.RadioItems(
                        id='morph',
                        options=[
                            {'label': 'Erosion', 'value': '1'},
                            {'label': 'Dilation', 'value': '2'},
                            {'label': 'Opening', 'value': '3'},
                            {'label': 'Closing', 'value': '4'},
                            {'label': 'None', 'value': '5'}
                        ],
                        value='5'
                    ),
                    html.Div(id='my-div-5')
            ])


@app.callback(
    Output(component_id='my-div-1', component_property='children'),
    [Input('upload-data', 'filename'), Input('blur', 'value'),
     Input('thresh', 'value'), Input('morph', 'value'),
     Input('upload-image', 'contents')],
    [State('upload-image', 'filename'),
     State('upload-image', 'last_modified')]
)
def update_output_div1(filename, blur_value, thresh_value, morph_value, list_of_contents, list_of_names, list_of_dates):
    print(UPLOAD_DIRECTORY)
    print(filename)
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
    else:
        children = []
    # print("Image: {}".format(children))
    data = list_of_contents[-1].encode("utf8").split(b";base64,")[1]

    # with open(os.path.join(UPLOAD_DIRECTORY, list_of_names[0]), "wb") as fp:
    #     fp.write(base64.decodebytes(data))
    filename = list_of_names[-1]
    image_filename = UPLOAD_DIRECTORY + "/" + filename  # replace with your own image
    # encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    # image = cv2.imread(encoded_image.decode())

    # gray = cv2.cvtColor(image_filename, cv2.COLOR_BGR2GRAY)
    gray = cv2.imread(filename, 0)
    # gray = cv2.imread(, 0)
    kernel = np.ones((5, 5), np.uint8)
    if blur_value == '2':
        gray = cv2.medianBlur(gray, 3)
    if blur_value == '4':
        gray = cv2.blur(gray, (5, 5))
    if blur_value == '1':
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if blur_value == '3':
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    if blur_value == '5':
        gray = gray
    # check to see if we should apply thresholding to preprocess the
    # image
    if thresh_value == 1:
        gray = gray
    if thresh_value == 2:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    if thresh_value == 3:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)[1]
    if thresh_value == 4:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TRUNC)[1]
    if thresh_value == 5:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO)[1]
    if thresh_value == 6:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO_INV)[1]
    if thresh_value == 7:
        gray = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 115, 1)
    if thresh_value == 8:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 115, 1)
    if thresh_value == 9:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    if thresh_value == 10:
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if morph_value == '1':
        gray = cv2.erode(gray, kernel, iterations=1)
    if morph_value == '2':
        gray = cv2.dilate(gray, kernel, iterations=1)
    if morph_value == '3':
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    if morph_value == '4':
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    if morph_value == '5':
        gray = gray


    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    # text = pytesseract.image_to_string(Image.open(filename))

    # print(text)
    encoded_image = base64.b64encode(open(filename, 'rb').read())
    os.remove(filename)
    gc.collect()
    return html.Img(id='image', src='data:image/png;base64,{}'.format(
                                               encoded_image.decode()))


@app.callback(
    Output(component_id='my-div-2', component_property='children'),
    [Input('upload-data', 'filename'), Input('first-slider', 'value'),
     Input('second-slider', 'value'), Input('class-selector', 'value')]
)
def update_output_div2(filename, second_value, third_value, first_value):
    """Convert image to grayscale."""
    print(UPLOAD_DIRECTORY)
    print(filename)
    print("123")
    # load the COCO class labels our Mask R-CNN was trained on
    labelsPath = "mask-rcnn-coco/object_detection_classes_coco.txt"
    LABELS = open(labelsPath).read().strip().split("\n")

    # load the set of colors that will be used when visualizing a given
    # instance segmentation
    colorsPath = "mask-rcnn-coco/colors.txt"
    COLORS = open(colorsPath).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = "mask-rcnn-coco/frozen_inference_graph.pb"
    configPath = "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

    # load our Mask R-CNN trained on the COCO dataset (90 classes)
    # from disk
    print("[INFO] loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    # load our input image and grab its spatial dimensions
    # load the example image and convert it to grayscale
    image_filename = UPLOAD_DIRECTORY + "/" + filename  # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    image = cv2.imread(encoded_image.decode())
    image = cv2.imread(image_filename)
    # image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    # show timing information and volume information on Mask R-CNN
    print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
    print("[INFO] boxes shape: {}".format(boxes.shape))
    print("[INFO] masks shape: {}".format(masks.shape))

    # loop over the number of detected objects
    for i in range(0, boxes.shape[2]):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        if first_value is None:
            first_value = LABELS
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > second_value and LABELS[classID] in first_value:
            # clone our original image so we can draw on it
            clone = image.copy()

            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
            mask = (mask > third_value)

            # extract the ROI of the image
            roi = clone[startY:endY, startX:endX]

            # check to see if are going to visualize how to extract the
            # masked region itself

            # convert the mask from a boolean to an integer mask with
            # to values: 0 or 255, then apply the mask
            visMask = (mask * 255).astype("uint8")
            instance = cv2.bitwise_and(roi, roi, mask=visMask)

            # show the extracted ROI, the mask, along with the
            # segmented instance
            # cv2.imshow("ROI", roi)
            # cv2.imshow("Mask", visMask)
            # cv2.imshow("Segmented", instance)

            # now, extract *only* the masked region of the ROI by passing
            # in the boolean mask array as our slice condition
            roi = roi[mask]

            # randomly select a color that will be used to visualize this
            # particular instance segmentation then create a transparent
            # overlay by blending the randomly selected color with the ROI
            color = random.choice(COLORS)
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original image
            clone[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the image
            color = [int(c) for c in color]
            cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)

            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(clone, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # write the grayscale image to disk as a temporary file so we can
            # apply OCR to it
            filename = "{}.png".format(os.getpid())
            cv2.imwrite(filename, clone)
            # load the image as a PIL/Pillow image, apply OCR, and then delete
            # the temporary file
            # text = pytesseract.image_to_string(Image.open(filename))

            # print(text)
            encoded_image = base64.b64encode(open(filename, 'rb').read())
            os.remove(filename)
            gc.collect()
            return html.Img(id='image', src='data:image/png;base64,{}'.format(
                                                       encoded_image.decode()))


@app.callback(
    Output(component_id='my-div-3', component_property='children'),
    [Input('upload-data', 'filename'), Input('first-slider', 'value'),
     Input('second-slider', 'value'), Input('class-selector', 'value')]
)
def update_output_div3(filename, second_value, third_value, first_value):
    """Convert image to grayscale."""
    print(UPLOAD_DIRECTORY)
    print(filename)
    print("321")
    """Convert image to grayscale."""
    # load the COCO class labels our YOLO model was trained on
    labelsPath = "yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = "yolo-coco/yolov3.weights"
    configPath = "yolo-coco/yolov3.cfg"

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image_filename = UPLOAD_DIRECTORY + "/" + filename  # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    image = cv2.imread(encoded_image.decode())
    image = cv2.imread(image_filename)
    # image = cv2.imread(args["image"])
    # image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]
            if first_value is None:
                first_value = LABELS
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > second_value and LABELS[classID] in first_value:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    print(boxes, confidences, classIDs)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, second_value,
                            third_value)

    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolo-coco/yolov3.cfg"
    modelWeights = "yolo-coco/yolov3.weights"

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    known_width = 0.6
    known_height = 1.8
    known_distance = 1.0
    focal_length = 346.0

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # print("Width {}".format(w))
            # print("Height {}".format(h))
            new_width = w
            new_height = h
            new_distance = (new_width*focal_length)/known_width
            # print("Focal length: {}".format(focal_length))
            area = w*h
            meters = ((known_width*focal_length)/new_width) + ((known_height*focal_length)/new_height)/2

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f} {:.2f}m".format(LABELS[classIDs[i]], confidences[i], meters)
            cv2.circle(image, (int(x+w/2), int(y+h*0.1)), 7, (255, 255, 255), -1)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    text = "{} detected".format(len(idxs))
    cv2.putText(image, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, image)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    # text = pytesseract.image_to_string(Image.open(filename))

    # print(text)
    encoded_image = base64.b64encode(open(filename, 'rb').read())
    os.remove(filename)
    gc.collect()
    return html.Img(id='image', src='data:image/png;base64,{}'.format(
                                               encoded_image.decode()))

'''
@app.callback(
    Output(component_id='my-div-4', component_property='children'),
    [Input('upload-data', 'filename'), Input('first-slider', 'value'),
     Input('second-slider', 'value'), Input('class-selector', 'value')]
)
'''
def update_output_text(filename, first_value, second_value, thresh, resize):
    """Convert image to grayscale."""
    # load the example image and convert it to grayscale
    print(UPLOAD_DIRECTORY)
    print(filename)
    image_filename = UPLOAD_DIRECTORY + "/" + filename  # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    image = cv2.imread(encoded_image.decode())
    # Resize to 2x

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image_filename, cv2.COLOR_BGR2GRAY)
    gray = cv2.imread(image_filename, 0)
    if resize != 1.0:
        gray = cv2.resize(gray, None,
                          fx=float(resize), fy=float(resize),
                          interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((5, 5), np.uint8)
    if first_value == '2':
        gray = cv2.medianBlur(gray, 3)
    if first_value == '4':
        gray = cv2.blur(gray, (5, 5))
    if first_value == '1':
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if first_value == '3':
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    if first_value == '5':
        gray = gray
    # check to see if we should apply thresholding to preprocess the
    # image
    if thresh == 1:
        gray = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if thresh == 2:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    if thresh == 3:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)[1]
    if thresh == 4:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TRUNC)[1]
    if thresh == 5:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO)[1]
    if thresh == 6:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO_INV)[1]
    if thresh == 7:
        gray = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 115, 1)
    if thresh == 8:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 115, 1)
    if thresh == 9:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    if thresh == 10:
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if second_value == '1':
        gray = cv2.erode(gray, kernel, iterations=1)
    if second_value == '2':
        gray = cv2.dilate(gray, kernel, iterations=1)
    if second_value == '3':
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    if second_value == '4':
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    if second_value == '5':
        gray = gray


    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    # text = pytesseract.image_to_string(Image.open(filename))
    text = pytesseract.image_to_string(gray)
    encoded_image = base64.b64encode(open(filename, 'rb').read())
    os.remove(filename)
    gc.collect()
    return text


@app.callback(Output('my-div-4', 'children'),
              [Input('upload-data', 'filename'),
               Input('blur', 'value'),
               Input('morph', 'value'),
               Input('thresh', 'value'),
               Input('resize', 'value')])
def update_output_div4(filename, first_value, second_value, thresh, resize):
    """Convert image to grayscale."""
    # load the example image and convert it to grayscale
    print(UPLOAD_DIRECTORY)
    print(filename)
    image_filename = UPLOAD_DIRECTORY + "/" + filename  # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    image = cv2.imread(encoded_image.decode())
    # Resize to 2x

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(image_filename, cv2.COLOR_BGR2GRAY)
    gray = cv2.imread(image_filename, 0)
    if resize != 1.0:
        gray = cv2.resize(gray, None,
                          fx=float(resize), fy=float(resize),
                          interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((5, 5), np.uint8)
    if first_value == '2':
        gray = cv2.medianBlur(gray, 3)
    if first_value == '4':
        gray = cv2.blur(gray, (5, 5))
    if first_value == '1':
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if first_value == '3':
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    if first_value == '5':
        gray = gray
    # check to see if we should apply thresholding to preprocess the
    # image
    if thresh == 1:
        gray = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if thresh == 2:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    if thresh == 3:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)[1]
    if thresh == 4:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TRUNC)[1]
    if thresh == 5:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO)[1]
    if thresh == 6:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO_INV)[1]
    if thresh == 7:
        gray = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 115, 1)
    if thresh == 8:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 115, 1)
    if thresh == 9:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    if thresh == 10:
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if second_value == '1':
        gray = cv2.erode(gray, kernel, iterations=1)
    if second_value == '2':
        gray = cv2.dilate(gray, kernel, iterations=1)
    if second_value == '3':
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    if second_value == '4':
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    if second_value == '5':
        gray = gray


    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    # text = pytesseract.image_to_string(Image.open(filename))

    # print(text)
    encoded_image = base64.b64encode(open(filename, 'rb').read())
    os.remove(filename)
    gc.collect()
    return html.Img(id='image', src='data:image/png;base64,{}'.format(
                                               encoded_image.decode()))


@app.callback(
    Output(component_id='my-div-5', component_property='children'),
    [Input('upload-data', 'filename'), Input('blur', 'value'),
     Input('thresh', 'value'), Input('morph', 'value'),
     Input('h1', 'value'), Input('s1', 'value'), Input('v1', 'value'),
     Input('h2', 'value'), Input('s2', 'value'), Input('v2', 'value')]
)
def update_output_div5(filename, blur_value, thresh_value, morph_value,
                       h1, s1, v1, h2, s2, v2):
    print(UPLOAD_DIRECTORY)
    print(filename)
    image_filename = UPLOAD_DIRECTORY + "/" + filename  # replace with your own image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    image = cv2.imread(encoded_image.decode())
    frame = cv2.imread(filename)
    # gray = cv2.cvtColor(image_filename, cv2.COLOR_BGR2GRAY)
    # frame = cv2.imread(image_filename)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([h1, s1, v1])
    upper_blue = np.array([h2, s2, v2])
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(np.array(hsv), lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    gray = res

    kernel = np.ones((5, 5), np.uint8)
    if blur_value == '2':
        gray = cv2.medianBlur(gray, 3)
    if blur_value == '4':
        gray = cv2.blur(gray, (5, 5))
    if blur_value == '1':
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if blur_value == '3':
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
    if blur_value == '5':
        gray = gray
    # check to see if we should apply thresholding to preprocess the
    # image
    if thresh_value == 1:
        gray = gray
    if thresh_value == 2:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    if thresh_value == 3:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)[1]
    if thresh_value == 4:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TRUNC)[1]
    if thresh_value == 5:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO)[1]
    if thresh_value == 6:
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO_INV)[1]
    if thresh_value == 7:
        gray = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 115, 1)
    if thresh_value == 8:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 115, 1)
    if thresh_value == 9:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    if thresh_value == 10:
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if morph_value == '1':
        gray = cv2.erode(gray, kernel, iterations=1)
    if morph_value == '2':
        gray = cv2.dilate(gray, kernel, iterations=1)
    if morph_value == '3':
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    if morph_value == '4':
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    if morph_value == '5':
        gray = gray


    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    # text = pytesseract.image_to_string(Image.open(filename))

    # print(text)
    encoded_image = base64.b64encode(open(filename, 'rb').read())
    os.remove(filename)
    gc.collect()
    return html.Img(id='image', src='data:image/png;base64,{}'.format(
                                               encoded_image.decode()))


def only_image(contents):
    gc.collect()
    return html.Img(id='image', src=contents)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    print("Name".format(name))
    print("Fuck".format(content.encode("utf8")))
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)


@app.callback(
    Output("file-list", "children"),
    [Input("upload-image", "filename"), Input("upload-image", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            print("Name here: ".format(name))
            print("Data here: ".format(data))
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]


def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('output-text', 'children'),
              [Input('upload-data', 'filename'),
               Input('blur', 'value'),
               Input('morph', 'value'),
               Input('thresh', 'value'),
               Input('resize', 'value')])
def calc_ocr(filename, first, second, thresh, resize):
    return 'You\'ve entered "{}"'.format(update_output_text(filename, first, second, thresh, resize))


if __name__ == '__main__':
    app.run_server(debug=True)
