from keras.applications import InceptionV3
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model


def InceptionV3SSENSE(nb_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
