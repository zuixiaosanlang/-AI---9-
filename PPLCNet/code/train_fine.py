import os
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.optimizer import lr
from model import PPLCNet
import random
import numpy as np
from paddle.optimizer import lr
from dataset1 import AngleClass1
from configs.config_base import Config
from visualdl import LogWriter
visual_log = LogWriter("training_log_refine")

seed = Config["solver"]["seed"]
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

num_epoch = Config["solver"]["num_epoch"]
save_path = Config["model"]["save_path"]
MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

train_dataset = AngleClass1(Config, 'train', False)
train_loader = paddle.io.DataLoader(train_dataset, batch_size = Config["dataset"]["train_batch"], shuffle=True, num_workers=0)

test_dataset=AngleClass1(Config, 'test', False)
test_loader = paddle.io.DataLoader(test_dataset,batch_size = Config["dataset"]["test_batch"], shuffle=False,num_workers=0)
print("train_dataset: ",len(train_dataset), "test_dataset: ",len(test_dataset))

net = PPLCNet(scale = Config["model"]["scale"], stages_pattern =MODEL_STAGES_PATTERN["PPLCNet"])
load_path = '../load/angleClass_80'
net.set_dict(paddle.load(load_path))
print("resume from {}".format(load_path))

max_accuracy = 0
schedule = lr.MultiStepDecay(learning_rate = Config["solver"]["base_lr"], milestones=Config["solver"]["milestones"], gamma = Config["solver"]["gamma"])
opt = paddle.optimizer.Adam(learning_rate=schedule, parameters=net.parameters())
save_dir = 'save'
with fluid.dygraph.guard():
    net.train()
    schedule.step()

    try:
        for epoch in range(0,Config["solver"]["num_epoch"]):
            
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
            model_state = net.state_dict()
            save_p = os.path.join(save_dir,'temp_epoch_refine')
            paddle.save(model_state,save_p)
            
            for  i, (img, labels) in enumerate(test_loader):
                predict = net(img)
                acc += fluid.layers.accuracy(input=predict,label=labels)
                total +=1
            visual_log.add_scalar(tag="eval_score", step=epoch, value = acc.numpy()/total)
            print("epoch {} /{} Accuracy {}".format(epoch, num_epoch, acc.numpy()/total))

            if acc.numpy()/total >max_accuracy:
                max_accuracy = acc.numpy()/total 
                model_state = net.state_dict()
                save_p = os.path.join(save_dir,'refine_best_model')
                paddle.save(model_state, save_p)
                print("max accuracy {}".format(max_accuracy))

            if epoch==64:
                #  最终提交模型为 temp 64
                save_p = os.path.join(save_dir,'temp_epoch_64')
                model_state = net.state_dict()
                paddle.save(model_state, save_p) 

    except KeyboardInterrupt:
        model_state = net.state_dict()
        paddle.save(model_state, Config["model"]["interrupt_path"])
