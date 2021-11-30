import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from utils import RANDOM_STATE
import cv2
import Data_Preprocessor as dp

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pd.set_option('display.max_columns', 5)


def predict(model, labelE):
    img_names = ['jayaram.jpg',
                 'mammooty.jpg',
                 'mohanlal.jpg',
                 'suresh_gopi.jpg']
    for i, img_name in enumerate(img_names):
        img_path = f'validation_images/{img_name}'
        img = cv2.imread(img_path)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_org = dp.validate_crop_img(img_gs, img)[0]
        tfmd_img = dp.wavelet_tfm(img_org, mode='db1', level=5)
        scaled_img_org = cv2.resize(img_org, (32, 32))
        scaled_tfm_img = cv2.resize(tfmd_img, (32, 32))
        combined_img = np.vstack((scaled_img_org.reshape(32 * 32 * 3, 1), scaled_tfm_img.reshape(32 * 32, 1)))
        x = np.array([combined_img]).reshape(1, 4096).astype(float)
        pred = labelE.inverse_transform(model.predict(x))[0]
        mimg = mpimg.imread(img_path)
        ax = plt.subplot(2,2, i+1)
        ax.patch.set_facecolor('xkcd:mint green')
        plt.imshow(mimg)
        real_name = img_name.split('.')[0]
        clr = 'green'
        if pred != real_name:
            clr = 'red'
        plt.title(f'actual: {real_name}; predicted:{pred}', c=clr)
    plt.show()


def train_model_and_predict(X_data, Y_data):
    X_data = np.array(X_data).reshape(len(X_data), 4096).astype(float)
    label_encoder = LabelEncoder()
    Y_labels = label_encoder.fit_transform(Y_data)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_labels, random_state=RANDOM_STATE)

    model_config = {
        'svm':{
            'model': SVC(gamma='auto', probability=True),
            'params':{
                'svc__C': [1, 5, 10],
                'svc__kernel': ['rbf', 'linear']
            }
        },
        'logistic_regression':{
            'model': LogisticRegression(solver='liblinear', multi_class='auto'),
            'params':{
                'logisticregression__C': [1, 5, 10]
            }
        }
    }

    scores = []
    best_estimators ={}
    for algo, mc in model_config.items():
        print(f'starting model {algo}\n')
        pipe = make_pipeline(StandardScaler(), mc['model'])
        clf_model = GridSearchCV(pipe, mc['params'], return_train_score=False)
        clf_model.fit(X_train, Y_train)
        scores.append({
            'model': algo,
            'best_score': clf_model.best_score_,
            'best_params': clf_model.best_params_
        })
        best_estimators[algo] = clf_model.best_estimator_

    print(pd.DataFrame(scores))

    print('svm score: ', best_estimators['svm'].score(X_test, Y_test))
    print('lr score: ', best_estimators['logistic_regression'].score(X_test, Y_test))

    print('\nclassification_report svm \n', classification_report(Y_test, best_estimators['svm'].predict(X_test)))
    print('\nclassification_report lr \n', classification_report(Y_test, best_estimators['logistic_regression'].predict(X_test)))

    print('\nconfusion_matrix svm: \n', confusion_matrix(Y_test, best_estimators['svm'].predict(X_test)))
    print('\nconfusion_matrix lr: \n', confusion_matrix(Y_test, best_estimators['logistic_regression'].predict(X_test)))

    print(predict(best_estimators['svm'], label_encoder))
