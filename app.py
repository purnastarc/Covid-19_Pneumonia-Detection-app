from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'COVID-19', 1 : 'Normal'}

model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(150,150))
	x = image.img_to_array(i)
	x=np.expand_dims(x,axis=0)
	images = np.vstack([x])
	classes = model.predict(images,batch_size=10)
	prediction='Intial'
	if classes==0:
		prediction='COVID-19'
	else:
		prediction='Normal'
	return prediction


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':

	app.run(debug = True)