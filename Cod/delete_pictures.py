import json
import os

### Iteram prin toate fiserele si stergem a 3-a poza, astfel daca vechia clasa era 3 acum va fi 0, mai exact cele doua poze ramase se asemeana, 
### iar daca vechia clasa era 1 sau 2 clasa devine 1 astfel inca pozele sunt diferite

with open('classes_updated.txt') as f:
    data = f.read()
js = json.loads(data)
print(js)
for i in js:
    if js[i] == 3:
        js[i] = 0
    else:
        js[i] = 1
print(js)
with open('classes_dual.txt', 'w') as convert_file:
     convert_file.write(json.dumps(js))

path = 'triplete_fete'
for i in os.listdir(path):
    os.remove(os.path.join(path,i,'3.jpg'))