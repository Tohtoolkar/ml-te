from flask import Flask, render_template, request, redirect, url_for, flash, session
#from flask_uploads import UploadSet, IMAGES, confiure_uploads
#from flask_wrtf import FlaskForm
#from flasl_wrf.file import FileField, FileRequired, FileAllowed
#from wtforms import SubmitField

#import io
from flask import Response

#from flask import Flask
#mport numpy as np

import pickle
import urllib.request, urllib.parse, urllib.error
import json
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


from collections import OrderedDict
from cbfv.composition import generate_features

import io
import base64

app = Flask(__name__)
model = pickle.load(open('model-TE3.pkl', 'rb'))


def merge_data(formula, sin_temp, measure_temp):
    
    table = pd.DataFrame(data = {'formula':[formula], 'sinter temp Celcius':[sin_temp], 'measurement temperature' : [measure_temp], 'target':[1]})
    
    data_and_features, y_train, formulae_train, skipped_train = generate_features(table, elem_prop='magpie', drop_duplicates=False, extend_features=True, sum_feat=True)
    
    data_and_selectFeatures = data_and_features[[
    'measurement temperature', 
    'mode_Number', 
    'avg_GSvolume_pa',
    'mode_GSvolume_pa', 
    'mode_Row', 
    #'range_NpValence', 
    #'range_NValence',
    'range_SpaceGroupNumber', 
    'sum_GSbandgap', 
    'dev_CovalentRadius',
    'sinter temp Celcius', 
    'max_NpUnfilled', 
    #'dev_Number',
    'avg_SpaceGroupNumber', 
    #'pressure MPa', 
    'range_NpUnfilled',
    'sum_CovalentRadius', 
    'max_MendeleevNumber', 
    'mode_Electronegativity',
    'mode_NUnfilled', 
    'dev_NdValence', 
    'dev_NpValence', 
    #'dev_NValence',
    'avg_NUnfilled', 
    'sum_Electronegativity',
    'sum_NValence',
    'sum_NsValence', 
    'mode_Column', 
    'dev_GSmagmom', 
    'sum_GSmagmom']]
    
    
    
    return data_and_selectFeatures
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER 

@app.route("/")
@app.route('/index')
def hello():
    return render_template('index.html' )


def Doped_mat():
    material = str(request.form['formula'])
    dopant1 = dopant1 = str(request.form['dopant1'])
    dopant2 = dopant2 = str(request.form['dopant2'])
    dop_con1 = float(request.form['dop_con1'])
    dop_con2 = float(request.form['dop_con2'])

  

    mat_host = {'h1': "", 'con1':"", 'h2':"", 'con2':"" , 'h3':""}
    if(material == "PbTe"):
        mat_host = {'h1': "Pb", 'con1':1, 'h2':"Te", 'con2':2 , 'h3':""}
    
    elif(material == "Co4Sb12"):
        mat_host = {'h1': "Co", 'con1':4, 'h2':"Sb", 'con2':12 , 'h3':""}
     
    elif(material == "Mg2Si"):
        mat_host = {'h1': "Mg", 'con1':2, 'h2':"Si", 'con2':1 , 'h3':""}
    
    elif(material == "BiCuSeO"):
        mat_host = {'h1': "Bi", 'con1':1, 'h2':"Cu", 'con2':1 , 'h3':"SeO"}
     
    elif(material == "Cu2Se"):
        mat_host = {'h1': "Cu", 'con1':2, 'h2':"Se", 'con2':1,  'h3':""}
    
     
    sum_dop1 = mat_host["con1"] - dop_con1
    sum_dop2 = mat_host["con2"] - dop_con2
    if(dop_con1 == 0.0):
        dopant1 = ""
        dop_con1 = ""
    else:
        dopant1 = str(request.form['dopant1'])
        dop_con1 = float(request.form['dop_con1'])
        sum_dop1 = mat_host["con1"] - dop_con1
        sum_dop2 = mat_host["con2"] - dop_con2   
  
    if(dop_con2 == 0.0):
        dopant2 = ""
        dop_con2 = ""
    else:
        dopant2 = str(request.form['dopant2'])
        dop_con2 = float(request.form['dop_con2'])
        sum_dop1 = mat_host["con1"] - dop_con1
        sum_dop2 = mat_host["con2"] - dop_con2
   

 

    pred_mat =  mat_host["h1"] + str(sum_dop1) + dopant1 + str(dop_con1) + mat_host["h2"] +str(sum_dop2) + dopant2 + str(dop_con2) + mat_host["h3"]
    print("pred_mat",pred_mat)
    return pred_mat

def generate_feature():
    formula = Doped_mat()
    sin_temp = int(request.form['sin_temp'])
    measure_temp = int(request.form['measure_temp'])
    featureTotal = merge_data(formula, sin_temp, measure_temp)

    
    
    return featureTotal

@app.route("/cal_zt", methods=['POST'])


def cal_zt():
    set_of_zt = []
    material = Doped_mat()
    sinter_temp = int(request.form['sin_temp'])
    #print("Doped_mat()", Doped_mat())
    set_of_zt = []
    temperatures = [500,550,600,650,700]
    set_of_zt = []
    for i in temperatures:
        zt = model.predict(merge_data(material,sinter_temp,i ))[0]
        set_of_zt.append(zt)
    
    return set_of_zt, temperatures, material,sinter_temp


def plot_zt2():
    zt_list = []

    set_of_zt, temperatures, material,sinter_temp = cal_zt()
    img=io.BytesIO()

   
    zt_list = set_of_zt
     
    plt.plot(temperatures, zt_list,  ms=9, mec='k', mfc='red', alpha=0.4, **{'color': 'lightsteelblue', 'marker': 'o'} )
    plt.xlim([470, 730])
    plt.ylim([0, 2])
    plt.title(f'Temperature dependent ZT ')
    plt.text(500, 1.85, f'Material: {material}', fontsize=10,  color='#3e424b',weight="bold")
    plt.text(500, 1.73, f'Sintered: {sinter_temp} $^๐$C', fontsize=10,  color='#3e424b',weight="bold")
    plt.xlabel(f' Temperature ($^๐$C)')
    plt.ylabel(f'ZT')
  
    
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = urllib.parse.quote(base64.b64encode(img.getvalue()).decode('utf-8'))

    plt.close()
    plt.cla()
    plt.clf()
    
    return plot_data

@app.route('/getdata/<string:predmat>', methods=['POST'])
def getdata(predmat):
    predmat = json.loads(predmat)
    
    print("----------")
    print(predmat)
    return redirect(url_for('predict',keys=predmat))

@app.route('/predict', methods=['POST',"GET"])
def predict(request):
    #formula = request.POST.get('formula')
    #dopant1 = request.POST.get('dopant1')
    #dopant2 = request.POST.get('dopant2')
    #dop_con1 = request.POST.get('dop_con1')
    #dop_con2 = request.POST.get('dop_con2')

    #formula = str(request.form['formula'])
    #sin_temp = int(request.form['sin_temp'])
    #measure_temp = int(request.form['measure_temp'])
    #feature = merge_data(formula, sin_temp, measure_temp)
   # data= json.loads(materialPred)
   # print("testtt", getdata(""))
   # print("------Check", getdata()
    try:
       
        pic2 =plot_zt2()
        pic = ""
        error =""

    except:
        error = "You put the wrong fomula form, try again!"
        pic2 = ""
        pic = ""
        print("Errror")

    
   
        
    #formula = str(request.form['formula'])
   # sin_temp = int(request.form['sin_temp'])
    #measure_temp = int(request.form['measure_temp'])
   
    #x= generate_feature()
    #prediction = model.predict(x)
    #output = prediction[0]
    #Flask_Logo = os.path.join(app.config['UPLOAD_FOLDER'], f'prediction_{formula}_{sin_temp}.jpg')
    
    return render_template(
        'index.html', 
        pic=pic, 
        error=error, 
        image2=pic2, 


     


        )


if __name__ == "__main__":
    app.run()
    
    
    
    
    
    
    