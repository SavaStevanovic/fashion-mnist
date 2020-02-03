import numpy as np
import cv2
import time
import tensorflow as tf
cap = cv2.VideoCapture(0)
i=0
start_time = time.clock()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_times = []    
avg_frame_time = 0.033
reporting_period = 30 
model = tf.keras.models.load_model('./models/separable_resnet14/cp.ckpt')

labels_dic = {
    0:'T-shirt/top',
    1:'Trouser',
    2:'Pullover',
    3:'Dress',
    4:'Coat',
    5:'Sandal',
    6:'Shirt',
    7:'Sneaker',
    8:'Bag',
    9:'Ankle boot'
}


background_images = []
for _ in range(400):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    background_images.append(frame)
background_mask = np.mean(np.array(background_images), 0)
background_mask_size = min(background_mask.shape[:-1])
background_mask=background_mask[:background_mask_size,:background_mask_size]
mask = background_mask
while(True):
    ret, frame = cap.read()
    frame=frame[:background_mask_size,:background_mask_size]
    mask = np.abs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)-background_mask)
    cv2.normalize(mask,  mask, 0, 255, cv2.NORM_MINMAX)

    _, mask = cv2.threshold(mask,30,255,cv2.THRESH_BINARY)
   
    frame = cv2.bitwise_and(frame,frame,mask = mask.astype(np.uint8))
    
    img = frame[background_mask_size//8:7*background_mask_size//8,background_mask_size//8:7*background_mask_size//8]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX,mask =  mask[background_mask_size//8:7*background_mask_size//8,background_mask_size//8:7*background_mask_size//8].astype(np.uint8))
    img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow('img',cv2.resize(img, (560,560), interpolation = cv2.INTER_NEAREST))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)
    label = model.predict(img/255)[0]

    id = np.argmax(label)
    label_str = labels_dic[id].ljust(12)
    label_descriptor = ''
    if label[id]>0.7:
        label_descriptor = label_str +':'+ "{:3.1f}".format(label[id]*100)
    frame_time_descriptor = 'Frame time:' + "{:1.4f}".format(avg_frame_time)
    timed_image = cv2.putText(frame.astype(np.uint8), frame_time_descriptor + '    '+ label_descriptor, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255) ,2)
    
    start_point = (background_mask_size//8,background_mask_size//8)
    end_point = (7*background_mask_size//8,7*background_mask_size//8)
    timed_image = cv2.rectangle(timed_image, start_point, end_point, (0,255,0), 2) 

    cv2.imshow('frame',timed_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    end_time = time.clock()
    frame_times.append(end_time-start_time)
    start_time = end_time
    i+=1

    if i % reporting_period == 0:
        avg_frame_time = sum(frame_times)/len(frame_times)
        frame_times = []
        i=0

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()