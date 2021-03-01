import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pandas as pd
import socket, threading
from sklearn.preprocessing import RobustScaler

model = load_model("meas_model.h5")
data = pd.read_csv("meas_train_dataset.csv")
scaler = RobustScaler()
scaled_train_data = scaler.fit(data)
print("Ready")


def binder(client_socket, addr):
	print('Connected by', addr)
	try:
		while True:
			data = client_socket.recv(64)
			msg = data.decode()
			num1 = float(msg)

			data = client_socket.recv(64)
			msg = data.decode()
			num2 = float(msg)
			
			data = client_socket.recv(64)
			msg = data.decode()
			num3 = float(msg)

			data = client_socket.recv(64)
			msg = data.decode()
			num4 = float(msg)
			
			data = client_socket.recv(64)
			msg = data.decode()
			num5 = float(msg)

			data = client_socket.recv(64)
			msg = data.decode()
			num6 = float(msg)

			num_return = [[num1, num2, num3, num4, num5, num6]]
			scaled_input = scaler.transform(num_return)
			prediction = model.predict(scaled_input)[0][0]
			data = str(prediction).encode()
			client_socket.send(data)
			
	except:
		print("except : " , addr);
	finally:
		client_socket.close();
		
			
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);
server_socket.bind(('', 9998));
server_socket.listen(1);


try:
	while True:
		client_socket, addr = server_socket.accept();
		th = threading.Thread(target=binder, args = (client_socket,addr));
		th.start();
except:
	print("server");
finally:
	server_socket.close();

