from cProfile import label
from decimal import ROUND_HALF_UP, ROUND_UP
from wsgiref.validate import validator
from flask import Flask, render_template, Response,jsonify,request,session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
from flask_bootstrap import Bootstrap
import cv2
import base64
import io
from utils.plots import plot_one_box

from hubconfCustom import video_detection
app = Flask(__name__)
Bootstrap(app)

app.config['SECRET_KEY'] = 'daniyalkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    # text = StringField(u'Conf: ', validators=[InputRequired()])
    conf_slide = IntegerRangeField('Confidence:  ', default=25,validators=[InputRequired()])
    submit = SubmitField("Run")
    



global results_id
global label_

def generate_frames(path_x = '',conf_= 0.25):
    yolo_output = video_detection(path_x,conf_)

    for detection_,corrdinates,color_,thickness in yolo_output:
        
        global results_id
        global label_
                
        
        

        for x in corrdinates:
            # image_ = 
    
            label_ = x[4]
            results_id = x[5]
            plot_one_box([int(x[0]),int(x[1]),int(x[2]),int(x[3])],detection_,label = x[4], color=color_,line_thickness=thickness)
        ref,buffer=cv2.imencode('.jpg',detection_)   
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')



@app.route("/",methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])

def home():
    session.clear()
    return render_template('root.html')


@app.route('/FrontPage',methods=['GET','POST'])
def front():
    # session.clear()
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        # conf_ = form.text.data
        conf_ = form.conf_slide.data
        
        # print(round(float(conf_)/100,2))
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        session['conf_'] = conf_
    return render_template('video.html',form=form)


@app.route('/video')
def video():
    return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results', methods=['GET'])
def fps_fun():
    global results_id
    global label_
    return jsonify(results_id=results_id,label_=label_)









if __name__ == "__main__":
    app.run(debug=True)
