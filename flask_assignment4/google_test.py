from PIL import Image
import numpy as np
from google_pred import predict_uva_landmark
import os

credential_path = 'lucid-honor-295522-305e6cacd6aa.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path


img = Image.open('0000002780.jpg')
img = img.resize((224,224))
img_array = np.array(img)
print(img_array.shape)
img_array = np.expand_dims(img_array,0)
print(img_array.shape)
print('Predicted Landmark: ' + str(predict_uva_landmark(img_array)))
		
