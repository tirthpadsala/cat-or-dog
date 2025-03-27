import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread     

model = pickle.load(open(r'C:\Users\padsa\tirth\python\codes\.ipynb_checkpoints\model.pkl', 'rb'))
# predicter = pickle.load(open(r'C:\Users\padsa\tirth\python\codes\.ipynb_checkpoints\predict_image.pkl', 'rb'))



iMage = open(r'C:\Users\padsa\tirth\python\csv_files\download.jpg','r')
def predict_image(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize

    prediction = model.predict(img_array)
    class_name = "Dog" if prediction[0][0] > 0.5 else "Cat"

    print(f"Predicted class: {class_name}, Confidence: {prediction[0][0]:.2f}")

imgage = imread(r'C:\Users\padsa\tirth\python\csv_files\download.jpg')
plt.imshow(imgage)
plt.show()
predict_image(r'C:\Users\padsa\tirth\python\csv_files\download.jpg')