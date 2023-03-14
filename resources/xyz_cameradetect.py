from PIL import Image, ImageOps
import sys
from keras.models import load_model
import numpy as np
import cv2

label = ''

def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)
        img_reshape = image[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction
    
def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

model = load_model('C:/Python/xyz/vgg16model_XYZ.hdf5') #or C:/Python/xyz/inceptionmodel_XYZ.hdf5
cap = cv2.VideoCapture(0)
make_720p()

if (cap.isOpened()):
    print("Camera OK")

while True:
    ret, original = cap.read()
    frame = cv2.resize(original, (224, 224))
    cv2.imwrite(filename='img.jpg', img=original)
    image = Image.open('img.jpg')
    # Display the predictions
    prediction = import_and_predict(image, model)
    if np.argmax(prediction) == 0:
        predict="It is X!"
    elif np.argmax(prediction) == 1:
        predict="It is Y!"
    else:
        predict="It is Z!"
    
    cv2.putText(original, predict, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0 , 0 , 255), 2)
    cv2.imshow("XYZ American Sign Language Classification", original)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()
