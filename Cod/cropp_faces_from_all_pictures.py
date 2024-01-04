import csv
import os
from crop_faces import cropp_picture


coordonate = []
with open('coordonate_fete.csv') as file_obj:
    reader_obj = csv.reader(file_obj)

    for row in reader_obj:
        coordonate.append(row)

lista_noua = []
for i in coordonate:
    lista_secundara = []
    for j in range(0,len(i),4):
        lista_secundara.append(i[j:j+4])
    lista_noua.append(lista_secundara)


lista_poze = []
for i in os.listdir('faces'):
    lista_poze.append(int(i))

lista_poze.sort()
lista_cai = []
for k in lista_poze:
    lista_cai_secundara=[]
    for j in os.listdir(os.path.join('faces',str(k))):
        lista_cai_secundara.append(os.path.join('faces',str(k),j))
    lista_cai.append(lista_cai_secundara)
print(len(lista_cai))



for i in range(5250, 11001):

    os.mkdir(os.path.join('faces_cropped_v2',str(i+1)))
    for j in range(2):

        try:
            cropp_picture(lista_cai[i][j],lista_noua[i][j],os.path.join('faces_cropped_v2',str(i+1),str(j+1)+'.jpg'))
        except:
            print(f'Poza din fisierul {i} cu numarul {j} nu a putut fi decupata')



