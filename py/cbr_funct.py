import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors
import pickle

from sklearn.preprocessing import MinMaxScaler

class CBRKNN:
  def __init__(self,train = None,target = None):
    self.modelKNN = None
    self.modelRF  = None
    self.train    = None
    self.index    = None
    self.trainnc  = train

    if train is None:
      self.modelKNN = pickle.load(open("modelKnn.sav", 'rb'))
      self.modelRF  = pickle.load(open("modelLinReg.sav", 'rb'))
      self.train    = pd.read_csv("dataCleanTrain.csv")
      self.index    = pd.read_csv("Data Corelation Result.csv",index_col="Unnamed: 0")
      self.trainnc  = pd.read_csv("py/data.csv")
    else :
      self.trainnc  = train
      df_new = self.preprocessingData(train,target)
      self.train,self.modelKNN,self.modelRF = self.training(df_new,target)

  def convert_val(self,x):
    if x<=10:
      return 0
    else:
      return 1

  def preprocessingData(self,df,target):
    df = df.dropna()
    df_object = df.select_dtypes(include='object').copy()
    df_number = df.select_dtypes(exclude='object').copy()

    df_object = pd.get_dummies(df_object) #menciptakan data dummy, karena data merupakan kategori
    df_number = df_number.fillna(df_number.median()) #kalau datanya kosong dengan median

    df_new = pd.concat([df_object,df_number],axis=1) #penggabungan data
    df_new.to_csv("After Preprocessing.csv")
    return df_new

  def training(self,df_new,target):
    most_correlated = df_new.corr()[target].sort_values(ascending=False)
    def columns_use(x):
      if x>0.05:
        return x
    self.index = most_correlated.apply(columns_use).dropna()
    df_use = df_new.loc[:, self.index.index]

    #Normalisasi Data Training
    scaler = MinMaxScaler()
    df_use.iloc[:,1:] = scaler.fit_transform(df_use.iloc[:,1:].values)
    
    self.index.to_csv("Data Corelation Result.csv") #Fitur yang digunakan

    # Konversi kelas
    df_use[target] = df_use[target].apply(self.convert_val) #Merubah data 20 class menjadi 2 class

    # Bagi fitur dan Membagi Dataset
    X = df_use.iloc[:,1:].values
    Y = df_use[target].values
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size =0.2)

    #Membangun model KNN
    knn2 = neighbors.KNeighborsClassifier() 
    leaf_size = list(range(1,30))
    n_neighbors = list(range(3,50,2))
    p=[1,2]
    weights = ['distance','uniform']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p, algorithm = algorithm,weights=weights)
    knn_gscv = GridSearchCV(knn2, hyperparameters)
    knn_gscv.fit(x_train, y_train)

    #Menyimpan Model
    pickle.dump(knn_gscv, open("modelKnn.sav", 'wb'))

    #Menyimpan data clean
    df_use.to_csv("dataCleanTrain.csv",index=False)

    print("Akurasi Train KNN     =",knn_gscv.best_score_*100)
    print("Akurasi Test KNN      =",metrics.accuracy_score(y_test,knn_gscv.predict(x_test)))

    return df_use,knn_gscv

  def run(self,data):
    #
    dataAdd = self.trainnc.drop(columns=['G3']).copy() #Mengambil Data Mentah digabung dengan data baru
    data = pd.concat([dataAdd,data],axis=0).reset_index(drop=True)
    
    #Preprocessing 
    data = self.preprocessingData(data,'G3')
    scaler = MinMaxScaler()
    temp = scaler.fit_transform(data.values)

    data = pd.DataFrame(temp, columns=list(data.columns))
    data = data.iloc[len(dataAdd):,:] 
    
    #Seleksi Fitur
    dataPred = [x for x in self.index.index if x!="G3"]

    data = data.loc[:, dataPred]
    dataBefore = data.copy()

    self.Retrieve(data.values)

    agree,hPred = self.Retrieve(data.values)
    ketData = ""

    if agree == 1:
      data = self.Reuse(data,hPred)
      print("Its Reuse")
      ketData = "Its Reuse"
    else:
      hPred = self.Revise(data)
      data = self.Reuse(data,hPred)
      print("Its Revise")
      ketData = "Its Revise"

    print("Prediction     :", hPred)
    return hPred,ketData

  def Retrieve(self,data):
    DataCheck = self.train.copy()
    DataCheck = DataCheck.drop(columns="G3")

    A = DataCheck.values
    B = data.flatten()

    cosine = np.dot(A,B)/(norm(A, axis=1)*norm(B))
    
    if cosine.max() > 0.9:
      return 1,(self.train.iloc[[np.argmax(cosine)]]['G3'].values[0])
    else:
      return 0,0

    if 0.9999999999999999 > cosine.max():
      print("Its New Data")
      dataInput = pd.DataFrame(data.values,columns=list(self.train.columns))
      self.train = pd.concat([self.train,dataInput],axis=0)
      self.train.to_csv("dataCleanTrain.csv", index=False)

  def Revise(self,data):
    return list(self.modelKNN.predict(data))[0]

  def Reuse(self,data,h):
    print(h)
    data['G3'] = h
    return data