import tensorflow as tf
# Define dataset paths
training_set = r'C:\Users\padsa\tirth\dog vs cat\dataset\training_set'
test_set = r'C:\Users\padsa\tirth\dog vs cat\dataset\test_set'

# Image augmentation for training data
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  
    rotation_range=40,    
    width_shift_range=0.2,  
    height_shift_range=0.2, 
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],  
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2
)

# Set parameters
target_size = (64, 64)
batch_size = 32

# Load training data
training_data = train_data_generator.flow_from_directory(
    training_set,  
    target_size=target_size,  
    batch_size=batch_size,
    shuffle=True,  
    subset='training',       
    class_mode='binary'
)

# Load validation data
validation_data = train_data_generator.flow_from_directory(
    training_set,  
    target_size=target_size,  
    batch_size=batch_size,
    shuffle=True,  
    subset='validation',  
    class_mode='binary'
)

# ImageDataGenerator for test data (no augmentation)
test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data = test_data_generator.flow_from_directory(
    test_set,  
    target_size=target_size,  
    batch_size=batch_size,
    shuffle=False,  
    class_mode='binary'
)

print("Data loaded successfully!")
import matplotlib.pyplot as plt
images , label = next(iter(test_data))

plt.figure(figsize=(10,10))

for i in range(1,10):
    plt.subplot(3,3 ,i)
    plt.imshow(images[i])
    plt.title(f'{label[i]}')
    plt.axis('off')
plt.show()
CNN= tf.keras.models.Sequential([
    tf.keras.layers.Conv2D( filters=64 , kernel_size=3 , activation='relu' , input_shape = (64, 64, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2) , strides=2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D( filters=128 , kernel_size=3 , activation='relu' ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2) , strides=2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D( filters=256 , kernel_size=3 , activation='relu' ),
    tf.keras.layers.MaxPool2D(pool_size=(2,2) , strides=2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(activation='relu' , units=256),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(activation='relu' , units=128),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(activation='sigmoid' , units=1)
])
CNN.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])
CNN.fit(x=training_data , validation_data=validation_data , epochs=20 , verbose=2 )

import numpy as np

def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize

    prediction = CNN.predict(img_array)
    class_name = "Dog" if prediction[0][0] > 0.5 else "Cat"

    print(f"Predicted class: {class_name}, Confidence: {prediction[0][0]:.2f}")

# Test with an image
predict_image(r'C:\Users\padsa\tirth\python\csv_files\cnn_test\dog2.jpg')
import pickle

pickle.dump(CNN ,open('model.pkl','wb'))
pickle.dump(predict_image ,open('predict_image.pkl','wb'))
f1 = pickle.load(open('model.pkl','rb'))
f2 = pickle.load(open('predict_image.pkl','rb'))

Cnn = f1
Predict_image = f2

Predict_image(r'C:\Users\padsa\tirth\python\csv_files\cnn_test\dog2.jpg')