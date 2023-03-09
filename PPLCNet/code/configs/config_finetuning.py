Config = {
    'dataset': {
        'base_dataset_path' :  '/home/aistudio/text_image_orientation',
        "base_ratio" : 10,
        "fine_ratio": 4,
        'base_train_label_path' : '/home/aistudio/text_image_orientation/train_list.txt',
        'base_test_label_path' : '/home/aistudio/text_image_orientation/test_list.txt',
        "aug_dataset_list": [
            '/home/aistudio/dataset/JPEGImages',
            '/home/aistudio/dataset/gt_blur',
        ],
        "aug_ratio_list": [3, 1],
        "train_batch": 256,
        "test_batch": 128, 
        "resize_pad": 320,
    },
    'solver':{
        'seed': 0,
        'base_lr':  0.5*1e-2,
        'milestones': [ 10, 20, 30, 50],
        'gamma': 0.5,
        'num_epoch': 50,
        'loss_print_freq': 10, 
    },
    'model':{
        'scale': 0.5,
        'pretrain_path':'/home/aistudio/work/angleClass_base_best',
        'save_path': 'angleClass_fine',
        'interrupt_path': 'angle-class-interrupt.pth',
    }
}