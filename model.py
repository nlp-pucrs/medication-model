import numpy as np
import pandas as pd
import warnings
import outliers
import csv
import pathlib
import itertools
import os 

def make_dir(direc):
    if not os.path.exists(direc):
        os.makedirs(direc)

warnings.filterwarnings('ignore')

file='prescriptions_sample.csv.gz' #add your csv data
prescription = pd.read_csv(file)
meds = prescription['medication']
meds = set(meds) #remove duplicates
meds = list(meds) #'listfies' set
meds = sorted(meds) #alphabetical order

Alpha=0.6
Metric='jaccard'

folder='modelos'


targs=[]

path=folder
make_dir(path)
for med_name in meds:
    ddc = outliers.ddc_outlier(alpha=Alpha,metric=Metric)
    med_dados, med_target = outliers.getPrescriptions(prescription,med_name)
    ddc.fit(med_dados)
    pr=ddc.pr #page rank
    
    pred_=ddc.predict(med_dados)
    pred = [0 if x==1 else 1 for x in pred_]

    dados=[]
    temp1=len(med_dados) # ==len(pred)
    for i in range(0,temp1):
        dados.append([med_dados[i][0],med_dados[i][1],pred[i]])
        targs.append(pred[i])
    dados.sort()
    dados = list(dados for dados,_ in itertools.groupby(dados))
    temp2=len(dados) #==len(pr)
    #path=str(Alpha)+'/'+folder+'/'
    #make_dir(path)
    with open(path+'/'+med_name+'.csv', 'w') as csvfile: #write data to folder
    
        writer = csv.writer(csvfile,delimiter=',')
        writer.writerow( ("Dose","Frequency", "Target","PageRank") )
        for i in range(0,temp2):
            writer.writerow([dados[i][0],dados[i][1],dados[i][2],pr[i]])
