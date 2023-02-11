from flask import Flask, render_template, request, url_for, redirect,session,jsonify
import pandas as pd
from py.cbr_funct import CBRKNN

app = Flask(__name__)
app = Flask(__name__,template_folder='temp')

cb = CBRKNN()
df = pd.read_csv('py/data.csv')

@app.route('/')
def index():
  return render_template("index.html")

@app.route('/cbr',methods=['POST'])
def cbr():
  data = request.form.to_dict(flat=False)

  data['G2'] = [int(x) for x in data['G2']]
  data['G1'] = [int(x) for x in data['G1']]
  data['Medu'] = [int(x) for x in data['Medu']]
  data['Fedu'] = [int(x) for x in data['Fedu']]
  data['studytime'] = [int(x) for x in data['studytime']]

  data = pd.DataFrame.from_dict(data)
  data = data.loc[:,list(df.iloc[:,:-1].columns)]
  print(data.head())
  pred,ket = cb.run(data)
  return jsonify(str(pred)+"_"+ket)

if __name__=='__main__':
  app.run()