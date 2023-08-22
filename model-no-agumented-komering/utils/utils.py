import cv2
import os
import json
import numpy as np
from glob import glob
from keras.models import load_model
from keras.utils import load_img, img_to_array, save_img


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_json_file(filename):
    data = None
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def write_or_append_json_file(filename, data):
    json_data = {}

    try:
        with open(filename, 'r') as temp:
            json_data = json.load(temp)

    except IOError:
        print('error')

    if 'data' in json_data: json_data['data'].append(data['data'][0])
    else: json_data = data

    with open(filename, 'w') as outfile:
        outfile.write(json.dumps(json_data))

# Define a sorting key function
def sort_by_top_left(contour):
    x, y, _, _ = cv2.boundingRect(contour)
    return (y // 10) * 10000 + x

def rect_contains(rect, pt):
    in_rect = rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
    return in_rect


def detect_bounding_box(
    image_path,
    save_threshold = False,
    save_threshold_path = None,
    erosion_size = 0,
    dilation_size = 0,
):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    if dilation_size > 0:
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        thresh = cv2.dilate(thresh, kernel, cv2.BORDER_REFLECT)

    if erosion_size > 0:
        kernel = np.ones((erosion_size, erosion_size), np.uint8)
        thresh = cv2.erode(thresh, kernel, cv2.BORDER_REFLECT)

    # Find contours, obtain bounding box, extract and save ROI
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.bitwise_not(thresh)

    if save_threshold:
        path_save = os.path.join(save_threshold_path, 'threshold')
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        cv2.imwrite("{}/threshold-white.png".format(path_save), image)
        cv2.imwrite('{}/threshold.png'.format(path_save), thresh)

    filtered_contours = sorted(cnts, key=sort_by_top_left)
    new_contours = []

    for i in range(len(filtered_contours)):
        current_contour = filtered_contours[i]
        is_inside = False

        for j in range(len(filtered_contours)):

            if j != i:
                other_contour = filtered_contours[j]

                if rect_contains(cv2.boundingRect(other_contour), tuple(current_contour[0][0])):
                    is_inside = True
                    break

        if not is_inside:
            new_contours.append(current_contour)

    return image, filtered_contours if len(new_contours) == 1 else new_contours


def crop_and_save_image(
    original_image,
    contours,
    path_save,
    threshold_padding = 15,
    min_width = 10,
    min_height = 10
):
    image = original_image.copy()
    num_detected_aksara = 0

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        if w > min_width and h > min_height:
            padded_x = max(x - threshold_padding, 0)
            padded_y = max(y - threshold_padding, 0)
            padded_w = min(w + (threshold_padding * 2), image.shape[1] - padded_x)
            padded_h = h + (threshold_padding * 2)

            detected_aksara = original_image[padded_y:padded_y+padded_h, padded_x:padded_x+padded_w]
            file_path_name = os.path.join(path_save, 'aksara_{}.png'.format(num_detected_aksara))

            cv2.imwrite(file_path_name, detected_aksara)

            # create data for store bounding box
            path_contour = os.path.join(path_save, 'data')
            if not os.path.exists(path_contour):
                os.makedirs(path_contour)

            contour_json = {
                "no": str(num_detected_aksara),
                "box": {
                    "w": padded_w,
                    "h": padded_h,
                    "x": padded_x,
                    "y": padded_y,
                }
            }

            contour_json_temp = [contour_json];
            contour_json_dict_temp = { 'data': contour_json_temp }


            path_file_json_contour = os.path.join(path_contour, 'data.json')
            write_or_append_json_file(path_file_json_contour, contour_json_dict_temp)

            num_detected_aksara += 1

    return num_detected_aksara, image,



def read_temp_by_upload_id(id, temporary_path):
    dir_path = os.path.join(temporary_path, id)
    return glob(dir_path + '/aksara_*.png')


def load_prediction_model(path_model):
    model_file_path = os.path.join(path_model, 'model.h5')
    return load_model(model_file_path)

def predict_detected_aksara(model, aksara, width=48, height=48):
    image = load_img(aksara, color_mode='grayscale', target_size=(width, height))
    input_arr = img_to_array(image)
    input_arr_np = np.array([input_arr])

    prediction = model.predict(input_arr_np)
    prediction_scores = np.array(prediction)

    predicted_label = np.argmax(prediction_scores[0])

    sorting = (-prediction).argsort()
    prob = prediction[0][sorting[0][0]] * 100
    prob = round(prob, 2)  # Keep it as a float

    probabilities = []  # Store the top probabilities
    for value in sorting[0][:5]:
        prob_value = prediction[0][value] * 100
        prob_value = round(prob_value, 2)  # Keep it as a float
        probabilities.append(prob_value)

    # replace image upload with image which has been predicted
    os.remove(aksara)
    save_img(aksara, input_arr)

    return predicted_label, prob, probabilities


def predict_aksaras(model, label, aksaras):
    results_prediction = []

    for aksara in aksaras:
        predicted, prob, prob_list = predict_detected_aksara(model, aksara)
        file_name = '/'.join(aksara.split('/')[-2:])

        results_prediction.append({
            'no': aksara.split('_')[-1].split('.')[0],
            'prediction': label[predicted][0],
            'score': float(prob),
            'image': '/assets/temp/{}'.format(file_name),
            'probabilities': prob_list  # Include probabilities in the results
        })

    return sorted(results_prediction, key=lambda x: int(x['no']))


def set_font_scale(image, font_scale_factor=0.002):
    height, width = image.shape[:2]
    font_scale = max(width, height) * font_scale_factor
    return font_scale

def generate_color(score):
    score = float(score)  # Convert score to float if it's not already
    if score > 80:
        color = (0, 225, 0)
    elif score > 50:
        color = (225, 0, 0)
    else:
        color = (0, 0, 255)
    return color