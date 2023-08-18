import cv2
import requests
import os
import base64
import pickle


def img2vec(img):
    resized_img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
    v, buffer = cv2.imencode(".jpg", resized_img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_data_string = "data:image/jpeg;base64," + img_str
    url = "http://localhost:8080/api/genhog"
    params = {"item_str": image_data_string}
    response = requests.get(url, params=params)
    return response.json()

# img = cv2.imread('../Cars Dataset/train/Audi/25.jpg')
# print(img2vec(img))

path = '../Cars Dataset/test/'
X = []
y = []
for sub in os.listdir(path):
    for fn in os.listdir(os.path.join(path,sub)):
        img_file_name = os.path.join(path,sub)+"/"+fn
        X.append(cv2.imread(img_file_name)) #เก็บข้อมูลในlist x ข้อ4-a
        y.append(sub) #เก็บlabel ข้อ4-b

#ข้อ5 ใช้ endpoint ที่ทำพาทหนึ่ง
carvectors = []
#print(len(X)) ทั้งหมด3352ภาพ
for img in range(len(X)):
    res = img2vec(X[img])
    vec = list(res["HOG"])
    vec.append(y[img])
    carvectors.append(res)

# write_path = "hogvectors_test.pkl"
# pickle.dump(carvectors, open(write_path,"wb"))
# print("data preparation is done")
