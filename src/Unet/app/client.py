import os, json
import numpy as np
from flask import Flask, render_template
from PIL import Image

import sys
sys.path.append('.')

app = Flask(__name__)

APP_ROOT = os.getenv('APP_ROOT', '/infer')

@app.route('/')
@app.route('/index')
# @app.route(APP_ROOT, methods=['GET','POST'])
def infer():
    
    # load image and turn it into a json list
    image = np.asarray(Image.open('static/yorkshire_terrier.jpg')).astype(np.float32)
    data = {'image': image.tolist()}
    
    # capture request response
    response = app.response_class(
    response = json.dumps(data),
    status=200,
    mimetype='application/json'
    )
    
    full_filename = os.path.join('static', 'yorkshire_terrier.jpg')
    
    return render_template('index.html', user_image = full_filename)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)



# import requests
# from PIL import Image
# import numpy as np

# import sys
# sys.path.append('.')

# ENDPOINT_URL = "http://0.0.0.0:80/infer"


# def infer():
#     image = np.asarray(Image.open('resources/yorkshire_terrier.jpg')).astype(np.float32)
#     data = {'image': image.tolist()}
#     response = requests.post(ENDPOINT_URL, json = data)
#     response.raise_for_status()
#     print(response.json())
    
# if __name__ =="__main__":
#     infer()



