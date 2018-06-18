"""loads images to np.arrays

"""

from keras.preprocessing.image import img_to_array, load_img

img = load_img("C:\\magisterka_data\\dogscats\\train\\cats\\cat.0.jpg")

img = img_to_array(img)

print(img)


