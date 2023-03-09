比赛详情链接 https://aistudio.baidu.com/aistudio/competition/detail/702/0/introduction

整工程下载（包含训练数据集、预训练模型、提交paddle文件）
链接：https://pan.baidu.com/s/1AaeNSXbp-s4O4mFlO7dBbw 
提取码：asdf 
--来自百度网盘超级会员V5的分享

github上的缺少数据集、预训练模型、提交paddle文件，建议直接下载整个工程复现

一、整体方案：
	
	yolov5实例分割+表格方向分类；
	其中实例分割算法选用yolov5最新发布版本：v7.0 - YOLOv5 SOTA Realtime Instance Segmentation。https://github.com/ultralytics/yolov5/releases/tag/v7.0
	表格方向分类选用PP-LCNet，直接使用其baseline模型即可https://aistudio.baidu.com/aistudio/projectdetail/5398861
	
	
二、复现

	数据集、预训练模型等均包含在工程目录内部,相关目录路径均采用相对路径，已配置好在程序内部，配置好环境执行几个命令即可复现；
	
	1. 数据预处理；
	防止靠近图像边缘的表格分割不好，在原图4周补32厚度的0像素；
	数据集及标签生成脚本见 segment/data_process.py 中paddle2yolo()方法
	训练数据及标签均已处理好放于工程中
	
	2.训练yolov5分割模型
	python segment/train.py
	考虑模型大小及速度问题，这里选用yolov5s.pt模型进行迁移学习；训练生成遵从yolov5，相关数据及模型保存在runs目录.
	
	3.yolo模型转为paddle模型
	python export-seg.py
	会生成 best-seg_paddle_model目录文件,即为paddle模型文件，拷贝至submit_files文件夹覆盖即可
	
	4.进入目录打包所有文件提交
	cd submit_files
	zip -r submit_yolov5seg.zip ./

	附：表格方向分类选用PP-LCNet，直接使用其baseline模型 https://aistudio.baidu.com/aistudio/projectdetail/5398861
	训练步骤：
	1）到比赛详情下载训练数据集 https://aistudio.baidu.com/aistudio/competition/detail/702/0/introduction
		解压后记录图片数据集路径
    2）修改PPLCNet/code/configs/config_base.py中base_dataset_path改为自己的数据集路径
    3）执行脚本训练
	   cd PPLCNet/code
       python train_base.py
	4) PPLCNet/code/save/ 目录会生成angleClass_45模型
	
三、分数提升探讨
	
	1.yolo分割模型中的输出并不是直接的mask，后面的运算过程均基于pytorch环境进行的，这样的代码提取出来后不能直接提交否则会报错。为解决这个问题，本人将其改成基于numpy环境的提交；如果能将该部分直接放到模型内部，应该会更好；
	
	2.提交的模型是基于yolov5s.pt模型进行迁移学习来的，可以选用分割准确率更高的模型，比如yolo5x.pt，但模型体积会变大很多，需要显存大的机器来训练；训练出来的模型提交分数应该会更高；
	
	3.关于yolov5的调优还是有很多的，可以修改默认的超参数文件data\hyps\hyp.scratch-low.yaml
	
	4.里面关于目标检测的非极大值抑制算法non_max_suppression_np(),特别是conf_thres参数，数值过大，会漏表格目标，过小会把非表格图像检测出来，分别影响P,PR分数；
	
	5.PP-LCNet重训练，更深入训练；

