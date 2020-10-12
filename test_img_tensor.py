import requests
import json
import uuid
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


uuid_seen = list()  # time indexes already seen


def fetch_images(ip, port):
    response = requests.request(method='GET', url='http://%s:%s/fetch?type=image' % (ip, port))
    return response.json()


def send_msg(data, meta, ip, port):
    payload = {'data':data, 'meta':meta}
    response = requests.request(method='POST', url='http://%s:%s' % (ip, port), json=payload)
    print(response.url, response.ok, response.status_code, meta)


def convert_from_json(json_str):
    lists = json.loads(json_str)
    arr = np.array(lists)
    arr = arr / 255.0
    new_arr = np.expand_dims(arr, 0)
    tensor = tf.convert_to_tensor(new_arr)
    return tensor


def main():
    detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
    while True:
        time.sleep(1)
        try:
            images = fetch_images('127.0.0.1', 9999)  # TODO parameterize this
            #print(len(images))
            for img in images:
                #print(img)
                if img['meta']['uuid'] not in uuid_seen:
                    image_tensor = convert_from_json(img['data'])
                    detector_output = detector(image_tensor)
                    class_ids = detector_output["detection_classes"]
                    uuid_seen.append(img['meta']['uuid'])
                    meta = {'time':img['meta']['time'], 'type':'inference.sense.optical.object.detection.recognition', 'format':'TBD', 'service':svc_name, 'uuid':str(uuid.uuid4()), 'parent_uuid':img['meta']['uuid']}
                    send_msg(class_ids, meta, '127.0.0.1', 9999)
        except Exception as oops:
            print(oops)


if __name__ == '__main__':
    print('starting faster rcnn inception resnet v2 object detector')
    images = fetch_images('127.0.0.1', 9999)  # TODO parameterize this
    image_tensor = convert_from_json(images[0]['data'])
    detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
    detector_output = detector(image_tensor)
    class_ids = detector_output["detection_classes"]
    print(class_ids)