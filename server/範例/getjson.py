from flask import Flask
from flask import request
import base64
import os
from time import strftime
#from PIL import Image
#from base64 import decodestring



app = Flask(__name__)
@app.route('/upload', methods=['POST'])
def post():

    print(request.is_json)

    content = request.get_json()    
    print(content['photo'])
    
    imagestr = content['photo']
    imgdata=base64.b64decode(imagestr)
    
    filename=strftime("%Y%m%d%H%M%S")
    filepath=os.path.join('./img/'+filename)
    
    with open(filepath,'wb') as f:
        f.write(imgdata)


    return content['photo']

app.run(host='140.116.67.155', port=5000)




