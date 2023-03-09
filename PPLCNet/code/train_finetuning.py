import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.optimizer import lr
from model import PPLCNet
import random
import numpy as np
from paddle.optimizer import lr
from dataset import AngleClass
from configs.config_finetuning import Config
import os

seed = Config["solver"]["seed"]
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

num_epoch = Config["solver"]["num_epoch"]
save_path = Config["model"]["save_path"]
MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

train_dataset = AngleClass(Config, 'train', False)
train_loader = paddle.io.DataLoader(train_dataset, batch_size = Config["dataset"]["train_batch"], shuffle=True, num_workers=0)

test_dataset=AngleClass(Config, 'train', True)
test_loader = paddle.io.DataLoader(test_dataset,batch_size = Config["dataset"]["test_batch"], shuffle=False,num_workers=0)
print("train_dataset: ",len(train_dataset), "test_dataset: ",len(test_dataset))

net = PPLCNet(scale = Config["model"]["scale"], stages_pattern =MODEL_STAGES_PATTERN["PPLCNet"])
load_path = Config["model"]["pretrain_path"]

if os.path.exists(load_path):
    net.set_dict(paddle.load(load_path))
    print("loading base model")
else:
   print("The base model can not find")

max_accuracy = 0
schedule = lr.MultiStepDecay(learning_rate = Config["solver"]["base_lr"], milestones=Config["solver"]["milestones"], gamma = Config["solver"]["gamma"])
opt = paddle.optimizer.Adam(learning_rate=schedule, parameters=net.parameters())

with fluid.dygraph.guard():
    net.train()
    schedule.step()

    try:
        for epoch in range(Config["solver"]["num_epoch"]):
            net.train()
            schedule.step()
            for i, (img,  labels) in enumerate(train_loader):
                predict = net(img)
                loss = F.cross_entropy(input=predict, label= labels)
                avg_loss = paddle.mean(loss)
                avg_loss.backward()
                opt.minimize(avg_loss)
                net.clear_gradients()

                acc = fluid.layers.accuracy(predict,labels)
                if i % Config["solver"]["loss_print_freq"]==0:
                    print("epoch {}/{} iter {} loss: {} train_acc {}".format(epoch,num_epoch, i, avg_loss.numpy(), acc.numpy()))

            total, acc = 0, 0
            net.eval()
            for  i, (img, labels) in enumerate(test_loader):
                predict = net(img)
                acc += fluid.layers.accuracy(input=predict,label=labels)
                total +=1
            print("epoch {} /{} Accuracy {}".format(epoch, num_epoch, acc.numpy()/total))
            if acc.numpy()/total >max_accuracy:
                max_accuracy = acc.numpy()/total 
                model_state = net.state_dict()
                paddle.save(model_state, Config["model"]["save_path"])
                print("max accuracy {}".format(max_accuracy))
    
    except KeyboardInterrupt:
        model_state = net.state_dict()
        paddle.save(model_state, Config["model"]["interrupt_path"])
