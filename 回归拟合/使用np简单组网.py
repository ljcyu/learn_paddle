import paddle
import  numpy as np
from visualdl import LogWriter
import datetime
import os

date_str=datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H%M%S')
log_dir=os.path.join('./logs',date_str)
writer=LogWriter(logdir=log_dir)

print(paddle.__version__)
# Define the model input data
x = np.array([[1.0], [2.0], [3.0], [4.0]])
# Define the label
y = np.array([[2.0], [4.0], [6.0], [8.0]])
# Define the model
net = paddle.nn.Linear(1, 1)
# Define the optimizer
optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=net.parameters())
# Define the loss function
loss_fn = paddle.nn.MSELoss()
# Train the model
for i in range(100):
   y_pred = net(paddle.to_tensor(x,dtype='float32'))
   loss = loss_fn(y_pred,paddle.to_tensor(y,dtype='float32'))
   loss.backward()
   optimizer.step()
   optimizer.clear_grad()
   writer.add_scalar(tag='train/loss',value=loss.numpy()[0],step=i)
   print(loss.numpy()[0])

#%%
