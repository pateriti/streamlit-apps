from flask import Flask, request, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import yolov5

#Instantiate Flask
api = Flask(__name__)

mobilenet_model = load_model('./best_estimator/mobilenet_model.h5')

#Create the routes
@api.route('/',methods = ['GET'])
def hello():
    # return index.html landing page
    return render_template('index.html')

#route 2: accept input data
#Post method is used when we want to receive some data from the user
@api.route('/predict', methods = ['POST'])
def make_predictions():
    #Image Classifier
    #Get the data sent over the API
    imagefile = request.files['imagefile']
    image_path = './user_input_images/'+imagefile.filename
    imagefile.save(image_path)    
    test_image = image.load_img(image_path)
    
    #convert the image to a matrix of numbers to feed into model
    test_image = image.img_to_array(test_image) # 1st: convert loaded image to array
    
    #2nd: https://www.tensorflow.org/api_docs/python/tf/expand_dims 
    #(to add additional 4th dummy dimension for batch on top of height, width, channel for a color image, 
    #to meet Tensorflow's expected no. of dimensions for input image
    test_image = np.expand_dims(test_image, axis=0)
    
    #3rd: to pre-process inputs to be in the same format expected by MobileNetV2
    test_image = preprocess_input(test_image) 
    result = mobilenet_model.predict(test_image) 
    
    
    #object detection
    wgt = './best_estimator/'+'logov7.pt'
    model = yolov5.load(wgt)    
       
    #set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # set image
    img = image_path

    # perform inference
    results = model(img)
    
    # inference with larger input size
    results = model(img, size=1280)
    
    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # show detection bounding boxes on image, save to 'static' folder, set exist_ok to True to prevent creating duplicate folders
    results.save(save_dir='static/', exist_ok=True)
    
    return render_template('index.html', prediction = str(np.argmax(result)), image='/static/'+str(results.files[0]))

#Main function that actually runs the API!
if __name__ == '__main__':
    api.run(host='0.0.0.0', 
            debug=True, # Debug=True ensures any changes to inference.py automatically updates the running API
            port=int(os.environ.get("PORT", 8080))
           ) 
