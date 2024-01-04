import concurrent.futures
import time
import requests
import os
import csv



# Read the csv and put the links in a list
img_url = []
with open('train_partial.csv') as file_obj:
    # Create reader object by passing the file
    # object to reader method
    reader_obj = csv.reader(file_obj)

    # Iterate over each row in the csv
    # file using reader object

    for row in reader_obj:
        img_url.append(row)

t1 = time.perf_counter()

# Function that accesses the link and downloads the image
def download_pictures(img,folder,k):
    img_bytes = requests.get(img).content
    img_name = k
    img_name = f'{img_name}.jpg'
    with open (os.path.join('faces',folder,img_name),'wb') as img_file:
        img_file.write(img_bytes)
        print(f'{img_name} s-a descarcat')


# I use multithreading to download pictures faster
k=0
with concurrent.futures.ThreadPoolExecutor() as executor:
    for i in img_url:
        k+=1
        z=0
        os.mkdir(os.path.join('faces',str(k)))
        for j in i:
            z+=1
            executor.submit(download_pictures,j,str(k),str(z))

t2 = time.perf_counter()
print(f'Finished in {t2-t1} seconds')
