from imageai.Prediction import ImagePrediction
import os
import sys

from tensorflow.python.keras.preprocessing import image
execution_path = os.getcwd()

pic = sys.argv[1]
pic = str(pic) + '.jpg'


def find_if_real(img):
    os.chdir('images')
    if img in os.listdir(os.getcwd()):
        return 1
    else:
        return 0


def predict_the_pic():
    prediction = ImagePrediction()
    results = []

    prediction.setModelTypeAsInceptionV3()
    prediction.setModelPath(os.path.join(
        execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"))
    prediction.loadModel()

    predictions, probabilities = prediction.predictImage(
        os.path.join(os.getcwd(), pic), result_count=5)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)


if find_if_real(pic):
    predict_the_pic()
else:
    print(f'There is not a image with the name of {pic} in this folder')
