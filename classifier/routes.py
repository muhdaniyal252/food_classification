from classifier.forms import UserInput
from classifier import app
from flask import render_template,redirect,url_for
from classifier.predict import make_prediction
import secrets,os
from PIL import Image


img_size = 250

def saveImage(imageFile):
    randomHax = secrets.token_hex(6)
    _,fileExtention = os.path.splitext(imageFile.filename)
    fileName = randomHax + fileExtention
    filePath =os.path.join(app.root_path,'static','prediction_images',fileName)
    temp_path =os.path.join('prediction_images',fileName)
    image = Image.open(imageFile)
    image.save(filePath)
    return filePath,temp_path

@app.route('/',methods=['GET','POST'])
def home():
    form = UserInput()
    result = None
    image = None
    show = False
    if form.imageFile.data:
        path,image = saveImage(form.imageFile.data)
        result = make_prediction(path)
        show = True
    return render_template('index.html', title='Food Classifier', form=form, image=image, result=result, show=show)
