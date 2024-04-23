raw_text = open("wonderland.txt", "r", encoding="utf-8").read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
chars_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocb = len(chars)

datax = []
datay = []

for i in range(0, n_chars - 100):
    cin = raw_text[i: i + 100]
    cout = raw_text[i + 100]
    datax.append([chars_to_int[c] for c in cin])
    datay.append(chars_to_int[cout])

import numpy as np
datax = np.array(datax)
datay = np.array(datay)

datax = np.reshape(datax, (1198, 60, 1))

x = datax / n_vocb

from keras.utils import  to_categorical
y = to_categorical(datay)

#Create RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(100, 1)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(58, activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam")

#Training and callback
from keras.callbacks import ModelCheckpoint

saved_weights = "weights-{epoch:03d}-{loss:.4f}.hdf5"
mcp = ModelCheckpoint(saved_weights, monitor="loss", save_best_only=True)

model.fit(x, y, epochs=100, batch_size=128, callbacks=[mcp])

#Prediction
int_to_chars = dict((c, i) for i, c in enumerate(chars))
start = np.random(0, n_chars - 100 - 1)
sentence = datax[start]
print(''.join(int_to_chars[s] for c in sentence for s in c))

output = ""
for i in range(50):
    xt = np.reshape(sentence, (1,100,1))
    xt = xt / n_vocb
    result = model.predict(xt)
    index = np.argmax(result)
    chr = int_to_chars[index]
    output = output + chr
    sentence = np.append(sentence, index)
    sentence = sentence[1:len(sentence)]

print(output)