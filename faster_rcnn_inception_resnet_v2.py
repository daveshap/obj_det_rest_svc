import requests
import json
import time
import numpy
#import tensorflow_hub as hub


def fetch_images(ip, port):
    response = requests.request(method='GET', url='http://%s:%s/fetch?type=image' % (ip, port))
    return response.json()


def main():
    while True:
        time.sleep(1)
        try:
            images = fetch_images('127.0.0.1', 9999)
            print(len(images))
        except Exception as oops:
            print(oops)

if __name__ == '__main__':
    print('starting faster rcnn inception resnet v2 object detector')
    main()
    #detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
    #detector_output = detector(image_tensor)
    #class_ids = detector_output["detection_classes"]