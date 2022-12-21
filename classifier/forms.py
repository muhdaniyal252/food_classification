from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField,FileAllowed

class UserInput(FlaskForm):
    imageFile = FileField('Upload Image',validators=[FileAllowed(['jpg','png'])])
    submit = SubmitField('Predict')
