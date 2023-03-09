train_label_p = '/home/aistudio/work/PPLCNet/train_label.txt'
test_label_p = '/home/aistudio/work/PPLCNet/test_label.txt'
f_train = open(train_label_p,'w')
f_test = open(test_label_p,'w')
with open('/home/aistudio/work/PPLCNet/label.txt','r') as f:
    lines = f.readlines()
    for (lid,line) in enumerate(lines): 
        if lid%10==0:
            f_test.writelines(line)
        else:
            f_train.writelines(line)