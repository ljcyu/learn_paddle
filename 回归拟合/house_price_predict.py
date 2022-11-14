import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt

BUF_SIZE = 500
BATCH_SIZE = 20

# 用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)
# 用于测试的数据提供器，每次从缓存中随机读取批次大小的数据
test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)

# 用于打印，查看uci_housing数据
train_data = paddle.dataset.uci_housing.train();
sampledata = next(train_data())
print(sampledata)

# 定义一个全连接层
with fluid.dygraph.guard():
    linear = fluid.dygraph.Linear(13, 1, dtype="float32")

iter = 0;
iters = []
train_costs = []


def draw_train_process(iters, train_costs):
    title = "training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.grid()
    plt.show()


EPOCH_NUM = 10
for pass_id in range(EPOCH_NUM):  # 训练EPOCH_NUM轮
    # 开始训练并输出最后一个batch的损失值
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader迭代器
        with fluid.dygraph.guard():
            test_x = np.array([x[0] for x in data], np.float32)
            test_y = np.array([x[1] for x in data], np.float32)
            X = fluid.dygraph.to_variable(test_x)
            y = fluid.dygraph.to_variable(test_y)
            # 优化器选用SGD随机梯度下降，学习率为0.001.
            optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=linear.parameters())
            y_predict = linear(X)
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(cost)
            optimizer.minimize(avg_cost)
            train_cost = [avg_cost.numpy()]
        if batch_id % 10 == 0:
            print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0]))  # 打印最后一个batch的损失值
        iter = iter + 1
        iters.append(iter)
        train_costs.append(train_cost[0])

    # 开始测试并输出最后一个batch的损失值
    test_costs = []
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader迭代器
        with fluid.dygraph.guard():
            test_x = np.array([x[0] for x in data], np.float32)
            test_y = np.array([x[1] for x in data], np.float32)
            X = fluid.dygraph.to_variable(test_x)
            y = fluid.dygraph.to_variable(test_y)
            y_predict = linear(X)
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            test_cost = fluid.layers.mean(cost)
            test_costs = [test_cost.numpy()]

    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    print('Test:%d, Cost:%.5f' % (pass_id, test_cost))  # 打印平均损失