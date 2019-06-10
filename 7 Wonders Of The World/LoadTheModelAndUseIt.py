
# load and evaluate a saved model
from keras.models import load_model

#The trained model is loaded into the variable, model
model = load_model('WonderOfTheWorld.h5')

#code used to make the prediction
#3 images are included in the file that are, Image1, Image2, Image3 but you can include more images and test on them.
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('image2.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image= np.expand_dims(test_image, axis=0)
result= model.predict(test_image)

if result[0][0]>=0.5:
    prediction='China Itza'

if result[0][1]>=0.5:
   prediction='Christ The Redeemer Statue'
  
if result[0][2]>=0.5:
    prediction='Machu Picchu'
       
if result[0][3]>=0.5:
    prediction='Petra'
            
if result[0][4]>=0.5:
    prediction='Taj Mahal'
               
if result[0][5]>=0.5:
    prediction='The Great Wall Of China'
                
if result[0][6]>=0.5:
    prediction='The Roman Colosseum'   
    
print(prediction)