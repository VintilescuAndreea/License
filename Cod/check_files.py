import os
import json

path = 'faces_cropped_v2'
lista = []
for i in os.listdir(path):
    k = 0
    for j in os.listdir(os.path.join(path, i)):
        k += 1
    if k<2:
        lista.append(int(i))


print(len(lista))
lista.sort()
print(lista)

'''
# Open the original JSON file
with open('clase.json', 'r') as f:
    data = json.load(f)

# Delete the specified keys from the dictionary
for key in lista:
    if str(key) in data:
        del data[str(key)]

# Write the modified data to a new JSON file
with open('modified.json', 'w') as f:
    json.dump(data, f)
'''
import shutil
for j in os.listdir(path):
    if int(j) in lista:
        shutil.rmtree(os.path.join(path, j))
        print(os.path.join(path, j))