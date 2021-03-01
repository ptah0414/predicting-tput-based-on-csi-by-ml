# 산점도 행렬을 그리기 위해 seaborn 패키지를 설치합니다
!pip install seaborn

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


import IPython

!pip install -q -U keras-tuner
import kerastuner as kt

dataset_path = 'meas.csv'

data = pd.read_csv(dataset_path)

data.head()

"""판다스를 사용하여 데이터를 읽습니다."""

column_names = ['SINR_all_rx','sss_snr', 'sss_rsrp', 'sss_sinr', 'Estimated_SINR', 'Channel_Quality_Indicator', 'L1_rx_throughput_mbps']

raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)

dataset = raw_dataset.copy()

dataset.tail()

dataset_pop = dataset.copy()

num = 0
for i in range(len(dataset)):
  # tput이 0이거나 441보다 큰 경우, 해당 행을 제거합니다
  if dataset_pop.L1_rx_throughput_mbps[i] == 0 or dataset_pop.L1_rx_throughput_mbps[i] > 441:
    num += 1

    dataset_pop.SINR_all_rx.pop(i)
    dataset_pop.sss_snr.pop(i)
    dataset_pop.sss_rsrp.pop(i)
    dataset_pop.sss_sinr.pop(i)
    dataset_pop.Estimated_SINR.pop(i)
    dataset_pop.Channel_Quality_Indicator.pop(i)
    dataset_pop.L1_rx_throughput_mbps.pop(i)

dataset.SINR_all_rx = dataset_pop.SINR_all_rx.copy()
dataset.sss_snr = round(dataset_pop.sss_snr.copy(), 2)
dataset.sss_rsrp = round(dataset_pop.sss_rsrp.copy(), 2)
dataset.sss_sinr = round(dataset_pop.sss_sinr.copy(), 2)
dataset.Estimated_SINR = round(dataset_pop.Estimated_SINR.copy(), 2)
dataset.Channel_Quality_Indicator = dataset_pop.Channel_Quality_Indicator.copy()
dataset.L1_rx_throughput_mbps = round(dataset_pop.L1_rx_throughput_mbps.copy(), 2)

print(num)

dataset.tail()

"""### 데이터 정제하기

이 데이터셋은 일부 데이터가 누락되어 있습니다.
"""

dataset.isna().sum()

"""문제를 간단하게 만들기 위해서 누락된 행을 삭제하겠습니다."""

dataset = dataset.dropna()

sns.pairplot(data=dataset, x_vars=['SINR_all_rx', 'Estimated_SINR', 'Channel_Quality_Indicator'], y_vars='L1_rx_throughput_mbps', size=3)

sns.pairplot(data=dataset, x_vars=['sss_snr', 'sss_rsrp', 'sss_sinr'], y_vars='L1_rx_throughput_mbps', size=3)

"""### 데이터셋을 훈련 세트와 테스트 세트로 분할하기

이제 데이터를 훈련 세트와 테스트 세트로 분할합니다.

테스트 세트는 모델을 최종적으로 평가할 때 사용합니다.
"""

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

"""### 데이터 조사하기

전반적인 통계를 확인합니다
"""

train_stats = train_dataset.describe()
train_stats.pop("L1_rx_throughput_mbps")
train_stats = train_stats.transpose()
train_stats

"""특성과 레이블 분리하기

특성에서 타깃 값 또는 "레이블"을 분리합니다. 이 레이블을 예측하기 위해 모델을 훈련시킬 것입니다.
"""

train_labels = train_dataset.pop('L1_rx_throughput_mbps')
test_labels = test_dataset.pop('L1_rx_throughput_mbps')

"""### 데이터 스케일링


"""

import pandas as pd

dataframe = pd.DataFrame(train_dataset)
dataframe.to_csv("meas_train_dataset.csv", header=False, index=False)

print(train_dataset)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

scaled_train_data = scaler.fit_transform(train_dataset)
scaled_test_data = scaler.transform(test_dataset)

print(scaled_train_data)

"""## 모델"""

from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            #max_value=512,
                                            max_value=64,
                                            step=32),
                               activation='relu'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='mae',
        metrics=['mae']
        )
    return model

tuner = RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=30,
    executions_per_trial=1,
    directory='my_dir',
    )

tuner.search(scaled_train_data, train_labels,
             epochs=10,
             validation_data=(scaled_test_data, test_labels))

tuner.results_summary()

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)

history = model.fit(scaled_train_data, train_labels, epochs = 30, validation_data = (scaled_test_data, test_labels))

model.summary()


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

import matplotlib.pyplot as plt

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [L1_rx_throughput_mbps]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,50])
  plt.legend()

plot_history(history)

loss, mae = model.evaluate(scaled_test_data, test_labels, verbose=2)

print("테스트 세트의 평균 절대 오차: {:5.2f} Mbps".format(mae))

"""## 예측


"""

test_predictions = model.predict(scaled_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [L1_rx_throughput_mbps]')
plt.ylabel('Predictions [L1_rx_throughput_mbps]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 500], [-100, 500])

# 모델을 저장합니다
model.save("meas_model.h5")

from keras.models import load_model

model = load_model('meas_model.h5')
model.summary()

