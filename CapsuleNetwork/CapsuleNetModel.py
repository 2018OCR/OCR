from ChnRec.CapsuleLayer import *
from keras.models import Model
from keras.layers import *
from keras import backend as K

# CNN+Capsule
input_image = Input(shape=(65, 64, 1))
cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
cnn = AveragePooling2D((2, 2))(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
cnn = Reshape((-1, 128))(cnn)
capsule = Capsule(28, 16, 3, True)(cnn)
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)

model = Model(inputs=input_image, outputs=output)
model.compile(
    loss=lambda y_true, y_pred: y_true * K.relu(0.9 - y_pred) ** 2 + 0.5 * (1 - y_true) * K.relu(y_pred - 0.1) ** 2,
    optimizer='adam',
    metrics=['accuracy'])

# model.summary()
