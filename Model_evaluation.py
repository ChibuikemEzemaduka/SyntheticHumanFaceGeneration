import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from scipy.linalg import sqrtm

#Load data
real_images = np.load('images.npy')
attributes = np.load('attributes5.npy')

#Find IS
def Inception_score(input_imag):
    input_imag = (input_imag * 127.5) + 127.5
    imag = []
    for i in input_imag:
        imag1 = resize(i, (299,299,3), 0)
        imag.append(imag1)
    imag = np.array(imag)
    deeptest = InceptionV3()
    pred = deeptest.predict(preprocess_input(imag))
    incep_sco = []
    deg = np.floor(len(input_imag) / 10)
    for j in range(10):
        begin, finish = int(j * deg), int(j * deg + deg)
        pred1 = pred[begin:finish]
        pred2 = np.expand_dims(pred1.mean(axis=0), 0)
        diverge = pred1 * (np.log(pred1 + 1E-16) - np.log(pred2 + 1E-16))
        sumdiverge = diverge.sum(axis=1)
        averagediverge = np.mean(sumdiverge)
        rated = np.exp(averagediverge)
        incep_sco.append(rated)
    averr, stann = np.mean(rated), np.std(rated)
    return averr, stann

# Find FID
def FID(real,synt):
    synt = (synt * 127.5) + 127.5
    imag = []
    for i in synt:
        imag1 = resize(i, (299, 299, 3), 0)
        imag.append(imag1)
    imag2 = []
    for i in real:
        imag3 = resize(i, (299, 299, 3), 0)
        imag2.append(imag3)
    imag2 = np.array(imag2)
    imag = np.array(imag)
    deeptest = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    realpred = deeptest.predict(preprocess_input(imag2))
    fakepred = deeptest.predict(preprocess_input(imag))
    realm, reals = realpred.mean(axis=0), np.cov(realpred, rowvar=False)
    fakem, fakes = fakepred.mean(axis=0), np.cov(fakepred, rowvar=False)
    differ = np.sum((realm - fakem) ** 2.0)
    comm = sqrtm(reals.dot(fakes))
    if np.iscomplexobj(comm):
        comm = comm.real
    rating = differ + np.trace(reals + fakes - 2.0 * comm)
    return rating

#Function to sample real images with their appropriate attributes in order to calculate FID
def gen_realimages(real_images, attributes, condition, no_imag):
    lis = []
    igg = 0
    for i in range(len(real_images)):
        compa = np.reshape(attributes[i], [5,1]).astype('int32') == np.reshape(condition, [5,1]).astype('int32')
        if compa.all():
            lis.append(real_images[i])
        if len(lis) == no_imag:
            break
    print ("Number of compatible realimages is: ", len(lis))
    return np.array(lis).astype('float32')


#Evaluation Code
deepgen = tf.keras.models.load_model('trained_model')
deepgen.summary()
seed = 80
number_of_images = 1000
images = []
condition = np.random.randint(2, size=(5,1))
#condition =np.reshape(np.array([0,0,1,1,0]), [5,1])
condition = np.reshape(condition, [1,5,1])
print ('condition is: ',condition)

for i in range(number_of_images):
    Noise_z = tf.random.normal([1,100,1], 0, 1, seed=seed+i)
    synthetic_image = deepgen.predict([Noise_z, condition])
    synthetic_image = tf.reshape(synthetic_image, (64,64,3))
    images.append(synthetic_image)

#Calculate IS score
isavg, isstd = Inception_score(np.array(images))
print ("IS score is:   " + "average_ISaverage:   " + str(isavg))

#Calculate FID
realo = gen_realimages(real_images, attributes, condition, number_of_images)
fidrating = FID(realo,np.array(images))
print ("FID score is: ", fidrating)


#Plot scaled Fake generated Images
for i in range(100):
    plt.subplot(10, 10, 1 + i)
    diag = (np.array(images[i]) * 127.5) + 127.5
    plt.axis('off')
    plt.imshow(diag.astype('uint8'))
plt.savefig('100_Fake_Images.png', dpi=300)
plt.show()













