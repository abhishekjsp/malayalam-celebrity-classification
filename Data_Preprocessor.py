import cv2
import pathlib
import numpy as np
import pywt

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')


def wavelet_tfm(img, mode='haar', level=1):
    img_tfm = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_tfm = np.float32(img_tfm)
    img_tfm /= 255

    coeffs = pywt.wavedec2(img_tfm, mode, level=level)
    coeffs = list(coeffs)
    coeffs[0] *= 0

    img_tfm = pywt.waverec2(coeffs, mode)
    img_tfm *= 255
    img_tfm = np.uint8(img_tfm)

    return img_tfm


def validate_crop_img(img, img_org):
    ret = []
    faces = face_cascade.detectMultiScale(img)
    for (fx, fy, fw, fh) in faces:
        cropped_img = img[fy:fy+fh, fx:fx+fw]
        cropped_img_org = img_org[fy:fy+fh, fx:fx+fw]
        if len(eye_cascade.detectMultiScale(cropped_img)) > 1:
            # ret.append(cropped_img)
            ret.append(cropped_img_org)
    return ret


def clean_data(data_dir, target_dir):
    print('Executing clean_data..')
    data_path = pathlib.Path(data_dir)
    file_list = [f for f in data_path.resolve().glob('**/*') if f.is_file()]
    # print(*file_list, sep='\n')
    num = 0
    for img_path in file_list:
        parent_folder = img_path.parent.name
        dest_folder = f'{target_dir}/{parent_folder}'
        img_org = cv2.imread(str(img_path))
        img_gs = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        cropped_faces = validate_crop_img(img_gs, img_org)
        for cropped_face in cropped_faces:
            pathlib.Path(dest_folder).mkdir(parents=True, exist_ok=True)
            file_name = dest_folder + f'/{parent_folder}_{num}.png'
            cv2.imwrite(file_name, cropped_face)
            num += 1


def prepare_dataset(source_dir):
    print('Executing prepare_dataset..')
    X_data = []
    Y_data = []
    data_path = pathlib.Path(source_dir)
    file_list = [f for f in data_path.resolve().glob('**/*') if f.is_file()]
    for img_path in file_list:
        label = img_path.parent.name
        img_org = cv2.imread(str(img_path))
        tfmd_img = wavelet_tfm(img_org, mode='db1', level=5)
        scaled_img_org = cv2.resize(img_org, (32, 32))
        scaled_tfm_img = cv2.resize(tfmd_img, (32, 32))
        combined_img = np.vstack((scaled_img_org.reshape(32*32*3, 1), scaled_tfm_img.reshape(32*32, 1)))
        X_data.append(combined_img)
        Y_data.append(label)

    return X_data, Y_data
