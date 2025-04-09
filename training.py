from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
import json
from tensorflow.keras.callbacks import EarlyStopping

dataset_dir = r"C:\Users\johnw\OneDrive\Desktop\EmergingTech_AI\dataset"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

fruit_classes = list(train_generator.class_indices.keys())
with open('class_indices.json', 'w') as f:
    json.dump(fruit_classes, f)

base_model = MobileNetV2(input_shape=(100, 100, 3), include_top=False, weights='imagenet')

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(120, activation='relu'),
    Dense(len(fruit_classes), activation='softmax')
])
base_model.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=20, validation_data=val_generator)

model.save('fruit_recognition_model.h5')
