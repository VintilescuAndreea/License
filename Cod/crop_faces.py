import cv2
import os



# Function that gets the picture, thecoordinates and the save path
def cropp_picture(path,bounding_box2,path_to_save):
    img = cv2.imread(path)

    bounding_box = [float(i) for i in bounding_box2]
    #bounding_box = [0.142578, 0.266602, 0.065421, 0.297508]
    h, w, _ = img.shape
    startpoint = (int(bounding_box[0]*w), int(bounding_box[2]*h))
    endpoint = (int(bounding_box[1]*w), int(bounding_box[3]*h))

    #image = cv2.rectangle(img, startpoint, endpoint,color,thickness=1)

    cropped_image = img[startpoint[1]:endpoint[1],startpoint[0]:endpoint[0]]

    cv2.imwrite(os.path.join(path_to_save),cropped_image)




#cropp_picture('triplets2/1/1.jpg',lista)