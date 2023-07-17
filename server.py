#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import zmq
import numpy as np
from numpy import newaxis
from keras.models import load_model

WINDOW_SIZE = 500
EMG_CHANNELS = 2

model = load_model(
    r"/home/Hand-Gesture-Recognition-Using-ML-with-EMG-Sensors/checkpoints/40-0.992"
)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    #  Wait for next request from client
    message = socket.recv(WINDOW_SIZE * EMG_CHANNELS * 4)

    EMG1, EMG2 = list(), list()

    for i in range(WINDOW_SIZE):
        EMG1.append(int.from_bytes(message[4 * i : 4 * (i + 1)], byteorder="little"))
        EMG2.append(
            int.from_bytes(
                message[4 * (i + WINDOW_SIZE) : 4 * (i + WINDOW_SIZE + 1)],
                byteorder="little",
            )
        )

    temp = np.column_stack((EMG1, EMG2))
    temp = temp[newaxis, :, :]

    pred = model.predict(temp)
    output = np.int32(np.argmax(pred[0])).item()

    # print("EMG1: ", EMG1)
    # print("EMG2: ", EMG2)

    #  Send reply back to client
    a = output.to_bytes(4, byteorder="big")
    print("sended: ", a)
    socket.send(a)
