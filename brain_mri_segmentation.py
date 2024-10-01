import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import requests
import streamlit as st
from PIL import Image

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def normalize(image):
    return image / 255.0

def augment_data(train_images, train_masks):
    data_gen_args = dict(rotation_range=10, 
                         width_shift_range=0.1, 
                         height_shift_range=0.1, 
                         shear_range=0.2, 
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 42
    image_datagen.fit(train_images, augment=True, seed=seed)
    mask_datagen.fit(train_masks, augment=True, seed=seed)
    
    return image_datagen, mask_datagen

def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    return x

def attention_gate(x, g, inter_shape):
    theta_x = layers.Conv2D(inter_shape, (1, 1), strides=(2, 2))(x)
    phi_g = layers.Conv2D(inter_shape, (1, 1))(g)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3), strides=(2, 2), padding='same')(phi_g)
    add_xg = layers.add([upsample_g, theta_x])
    relu_xg = layers.Activation('relu')(add_xg)
    psi = layers.Conv2D(1, (1, 1), activation='sigmoid')(relu_xg)
    return layers.multiply([x, psi])

def nested_unet(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    final_layer = conv_block(inputs, 64)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(final_layer)
    return models.Model(inputs, outputs)

def attention_unet(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    final_layer = conv_block(inputs, 64)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(final_layer)
    return models.Model(inputs, outputs)

def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def load_data(data_path):
    return np.random.rand(100, 256, 256, 1), np.random.rand(100, 256, 256, 1)

def train_models():
    train_images, train_masks = load_data('./data')
    train_images = [apply_clahe(img) for img in train_images]
    train_images = [normalize(img) for img in train_images]
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)
    image_gen, mask_gen = augment_data(X_train, y_train)
    nested_model = nested_unet()
    nested_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])
    nested_model.fit(image_gen.flow(X_train, y_train, batch_size=16), epochs=50, validation_data=(X_test, y_test))
    attention_model = attention_unet()
    attention_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])
    attention_model.fit(image_gen.flow(X_train, y_train, batch_size=16), epochs=50, validation_data=(X_test, y_test))
    nested_model.save('nested_unet.h5')
    attention_model.save('attention_unet.h5')

app = FastAPI()

nested_model = None
attention_model = None

def load_models():
    global nested_model, attention_model
    nested_model = tf.keras.models.load_model("nested_unet.h5", compile=False)
    attention_model = tf.keras.models.load_model("attention_unet.h5", compile=False)

@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_type: str = 'nested'):
    load_models()
    image = await file.read()
    img = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=-1) / 255.0
    img = np.expand_dims(img, axis=0)
    if model_type == 'nested':
        prediction = nested_model.predict(img)
    else:
        prediction = attention_model.predict(img)
    return {"prediction": prediction.tolist()}

def run_streamlit_ui():
    st.title("Brain MRI Metastasis Segmentation")
    model_option = st.selectbox("Select a model:", ("Nested U-Net", "Attention U-Net"))
    uploaded_file = st.file_uploader("Choose a brain MRI image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        img_array = np.array(image.convert("L"))
        img_array = cv2.resize(img_array, (256, 256))
        model_type = 'nested' if model_option == 'Nested U-Net' else 'attention'
        response = requests.post("http://localhost:8000/predict/", files={"file": uploaded_file.getvalue()}, data={"model_type": model_type})
        if response.ok:
            prediction = np.array(response.json()["prediction"])
            st.image(prediction[0], caption='Segmented Metastasis', use_column_width=True)

if __name__ == "__main__":
    # train_models()
    # run_streamlit_ui()
    pass
