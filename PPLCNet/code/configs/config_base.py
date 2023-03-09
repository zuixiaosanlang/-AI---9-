Config = {
    'dataset': {
        'base_dataset_path': 'G:/work/for_fun/fei_jiang/table/Dataset/inpainting_kpt/train_A',
        "base_ratio": 1,
        "fine_ratio": 1,
        'base_train_label_path': '../train_label.txt',
        'base_test_label_path': '../test_label.txt',
        "aug_dataset_list": [
        ],
        "aug_ratio_list": [],
        "train_batch": 16,
        "test_batch": 16,
        "resize_pad": 624,
    },
    'solver': {
        'seed': 0,
        'base_lr': 0.5 * 1e-2,
        'milestones': [10, 20, 30, 40, 50, 60, 70],

        'gamma': 0.5,
        'num_epoch': 45,
        'loss_print_freq': 20,
    },
    'model': {
        'scale': 1,
        'pretrain_path': '',
        'save_path': 'angleClass_base',
        'interrupt_path': 'angle-class-interrupt.pth',
    }
}
