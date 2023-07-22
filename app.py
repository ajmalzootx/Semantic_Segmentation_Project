import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file,render_template
from tensorflow.keras.applications import resnet
from keras.models import load_model
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm



app = Flask(__name__)

# Your model and '/segment' route go here...

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights = weights)
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

def jaccard_coef(y_true, y_pred):
  y_true_flatten = K.flatten(y_true)
  y_pred_flatten = K.flatten(y_pred)
  intersection = K.sum(y_true_flatten * y_pred_flatten)
  final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
  return final_coef_value

@app.route('/segment', methods=['POST'])
def segment_image():
    # Get the uploaded image from thse request
    image_file = request.files['image']
    if not image_file:
        return jsonify({'error': 'No image provided'}), 400

    image = Image.open(image_file)
    image = image.resize((256,256))
    image = np.array(image)
    image = np.expand_dims(image, 0)

    
    # Perform semantic segmentation
    model = load_model('segment_model_munet.h5',
                         custom_objects=({'dice_loss_plus_1focal_loss': total_loss,
                                          'jaccard_coef': jaccard_coef}))
    # Replace this with your actual segmentation code
    prediction = model.predict(image)
    predicted_image = np.argmax(prediction, axis=3)
    predicted_image = predicted_image[0,:,:]
    segmented_image_new = Image.open('m1.png')

    image_np_rescaled = (predicted_image * 255).astype(np.uint8)
    segmented_image = Image.fromarray(image_np_rescaled)
    # Save the segmented image to a byte stream
    output_stream = io.BytesIO()
    segmented_image_new.save(output_stream, format='png')
    output_stream.seek(0)

    print(output_stream)

    return send_file(output_stream, mimetype='image/png')


# Add a route to serve the index.html file
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
