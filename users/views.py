from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import os


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def training(request):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    # Define paths
    train_dir = settings.MEDIA_ROOT + '//' + 'train'
    test_dir = settings.MEDIA_ROOT + '//' + 'test'

    # Image data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    # Load the VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)  # 4 classes

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the layers of VGG16
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
        epochs=10  # You can increase this number
    )

    # Save the model
    

    accuracy = history.history['accuracy'][-1]
    train_accuracy = history.history['accuracy']
    train_loss = history.history['loss']
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    print(f"Final training accuracy: {train_accuracy[-1]}")
    print(f"Final training loss: {train_loss[-1]}")
    print(f"Final validation accuracy: {val_accuracy[-1]}")
    print(f"Final validation loss: {val_loss[-1]}")

    print("Model training completed and saved as 'vgg_model.h5'")
    return render(request,'users/training.html',{'acc':accuracy})



import os
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'vgg_model.h5'), compile=False)

def predict_image(request):
    file_url = None
    predicted_class_name = None

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)

        # Save with original name first
        file_path = fs.save(image_file.name, image_file)
        original_image_path = os.path.join(settings.MEDIA_ROOT, file_path)

        # Convert TIFF to JPEG if necessary
        if file_path.lower().endswith('.tif') or file_path.lower().endswith('.tiff'):
            image = Image.open(original_image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Rename to JPG and save
            jpg_filename = os.path.splitext(file_path)[0] + '.jpg'
            jpg_image_path = os.path.join(settings.MEDIA_ROOT, jpg_filename)
            image.save(jpg_image_path, 'JPEG')

            # Delete original TIFF file after conversion
            os.remove(original_image_path)
        else:
            jpg_filename = file_path  # Keep original if it's already JPG/PNG
            jpg_image_path = original_image_path

        # Load and preprocess the image
        image = load_img(jpg_image_path, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        image_array = tf.expand_dims(image_array, 0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image_array)
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        class_names = ['Abnormal Detected "Bicycle"', 'Abnormal Detected "Electric_wheel"', 'Normal Detected "Skating"', 'Normal Detected "Van"']  # Replace with actual class names
        predicted_class_name = class_names[predicted_class]

        # URL for displaying image in template
        file_url = settings.MEDIA_URL + jpg_filename

    return render(request, 'users/predict.html', {
        'file_url': file_url,
        'predicted_class': predicted_class_name
    })
