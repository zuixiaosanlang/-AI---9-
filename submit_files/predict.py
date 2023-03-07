# 代码示例
# python predict.py [src_image_dir] [results]

import os
import sys
import glob
import json
import cv2

import paddle
import math

import cv2
import numpy as np
import glob
import os

from pplcnet.cls_model import PPLCNet


def resize_to_test(img, sz=(640, 480)):
    imw, imh = sz
    return cv2.resize(np.float32(img), (imw, imh), cv2.INTER_CUBIC)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个boundingbox的面积
    order = scores.argsort()[::-1]  # boundingbox的置信度排序
    keep = []  # 用来保存最后留下来的boundingbox
    while order.size > 0:
        i = order[0]  # 置信度最高的boundingbox的index
        keep.append(i)  # 添加本次置信度最高的boundingbox的index

        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留交集小于一定阈值的boundingbox
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    out = np.array(keep, dtype=int)
    return out


def non_max_suppression_np(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    mi = 5 + nc  # mask start index
    # output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output = [np.zeros((0, 6 + nm), dtype=np.float32)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            # v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v = np.zeros((len(lb), nc + nm + 5), dtype=np.float32)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            # x = torch.cat((x, v), 0)
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            # x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            # conf, j = x[:, 5:mi].max(1, keepdim=True)
            conf, j = np.max(x[:, 5:mi], 1)[:, np.newaxis], np.argmax(x[:, 5:mi], 1)[:, np.newaxis]
            # x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            x = np.concatenate((box, conf, j.astype(np.float32), mask), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            cls = x[:, 5:6].astype(np.int32).reshape(-1)
            sel_index = []
            for s in range(cls.shape[0]):
                if cls[s] in classes:
                    sel_index.append(s)
            x = x[sel_index]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            # x = x[x[:, 4].argsort(descending=True)]  # sort by confidence
            x = x[x[:, 4].argsort()[::-1]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def sigmoid_np(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))


def crop_mask_np(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    # x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    # r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    r = np.arange(w, dtype=np.float32)[None, None, :]
    # c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
    c = np.arange(h, dtype=np.float32)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_np(protos, masks_in, bboxes, shape, upsample=False):
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    # masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
    masks = sigmoid_np(masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask_np(masks, downsampled_bboxes)  # CHW
    if upsample:
        # masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        masks = cv2.resize(masks[0], (shape[1], shape[0]))
    # return masks.gt_(0.5)
    _, masks = cv2.threshold(masks, 0.5, 1, cv2.THRESH_BINARY)
    return masks[None]


def ResizePad(img, target_size):
    img = np.array(img)
    h, w = img.shape[:2]
    m = max(h, w)
    ratio = target_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    top = (target_size - new_h) // 2
    bottom = (target_size - new_h) - top
    left = (target_size - new_w) // 2
    right = (target_size - new_w) - left
    img1 = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img1


def process_image(img, mode, rotate):
    if mode == 'train':
        resize_width = 624
        img = ResizePad(img, resize_width)
    else:
        resize_width = 624
        img = ResizePad(img, resize_width)
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255

    return img


def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    test_h = 640
    test_w = 640

    import paddle.inference as pdi
    config = pdi.Config('best-seg_paddle_model/inference_model/model.pdmodel',
                        'best-seg_paddle_model/inference_model/model.pdiparams')
    config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
    predictor = pdi.create_predictor(config)
    input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    output_names = predictor.get_output_names()

    # paddle.disable_static()
    # params = paddle.load(r'pd_model/model.pdparams')
    # model = TFModel()
    # model.set_dict(params, use_structured_name=False)
    # model.eval()

    MODEL_STAGES_PATTERN = {
        "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
    }
    net = PPLCNet(scale=1, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"])
    load_path = 'pplcnet/angleClass_45'
    net.set_dict(paddle.load(load_path))
    net.eval()

    result = {}

    for image_path in image_paths:
        # image_path='G:/work/for_fun/fei_jiang/table_structure/yolov5-7.0/dataset/table-seg/images/border_0_0GEPLRUE92Y15IJ0O4B9.jpg'
        filename = os.path.split(image_path)[1]

        # do something
        im0 = cv2.imread(image_path)

        pad = 32
        h, w, c = im0.shape
        new_h, new_w = h + 2 * pad, w + 2 * pad
        img_pad = np.zeros(shape=(new_h, new_w, 3), dtype=np.uint8)
        img_pad[pad:pad + h, pad:pad + w] = im0

        src = img_pad.copy()
        im = cv2.resize(np.float32(img_pad), (test_w, test_h))

        im = im.astype(np.float32) / 255.0

        im2 = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im2 = np.ascontiguousarray(im2)[None]

        input_handle.copy_from_cpu(im2)
        predictor.run()
        model_out = [predictor.get_output_handle(x).copy_to_cpu() for x in output_names]
        pred = non_max_suppression_np(model_out[0], conf_thres=0.5, iou_thres=0.45, classes=None, nm=32)

        if filename not in result:
            result[filename] = []

        fx = src.shape[1] / test_w
        fy = src.shape[0] / test_h

        for i, det in enumerate(pred):
            if len(det):
                masks = process_mask_np(model_out[1][i], det[:, 6:],
                                        det[:, :4], im2.shape[2:], upsample=True)

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes([test_h, test_w], det[:, :4], im.shape).round()
                for i, det0 in enumerate(det):
                    x0, y0, x1, y1 = det0[:4]
                    conf = det0[4]
                    cls = det0[5]

                    boundingRect = [float(x0) * fx - pad, float(y0) * fy - pad, float(x1) * fx - pad,
                                    float(y1) * fy - pad]

                    boundingRect[0] = np.clip(boundingRect[0], 0, im0.shape[1])  # x1, x2
                    boundingRect[2] = np.clip(boundingRect[2], 0, im0.shape[1])  # x1, x2
                    boundingRect[1] = np.clip(boundingRect[1], 0, im0.shape[0])  # y1, y2
                    boundingRect[3] = np.clip(boundingRect[3], 0, im0.shape[0])  # y1, y2

                    ###
                    crop_img = im0[int(boundingRect[1]):int(boundingRect[3]), int(boundingRect[0]):int(boundingRect[2]), :]
                    crop_img = process_image(crop_img, 'test', True)
                    crop_img = paddle.to_tensor(crop_img)
                    crop_img = crop_img.unsqueeze(0)
                    with paddle.no_grad():
                        label = net(crop_img)
                    label = label.unsqueeze(0).numpy()
                    mini_batch_result = np.argsort(label)
                    mini_batch_result = mini_batch_result[0][-1]  # 把这些列标拿出来
                    mini_batch_result = mini_batch_result.flatten()  # 拉平了，只吐出一个 array
                    mini_batch_result = mini_batch_result[::-1]  # 逆序
                    pred_label = mini_batch_result[0]

                    ###
                    masks_in = masks[0][int(y0):int(y1), int(x0):int(x1)]
                    contours, _ = cv2.findContours(masks_in.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    max_size = 0
                    cnt_save = None
                    for cont in contours:
                        # 预测多边形
                        rect = cv2.minAreaRect(cont)
                        areas = min(rect[1])
                        if areas > max_size:
                            max_size = areas
                            cnt_save = cont
                    if cnt_save is not None:
                        rect = cv2.minAreaRect(cnt_save)
                        box = cv2.boxPoints(rect)
                        epsilon = 0.01 * cv2.arcLength(cnt_save, True)
                        box_dp = cv2.approxPolyDP(cnt_save, epsilon, True)
                        hull = cv2.convexHull(box_dp)
                        len_hull = len(hull)

                        if len_hull == 4:
                            # sort_points_clockwise(hull[:, 0, :], box)
                            box = hull[:, 0, :]

                        box[:, 0] = box[:, 0] + x0
                        box[:, 1] = box[:, 1] + y0

                        box[:, 0] = np.clip(box[:, 0], pad / fx, (im0.shape[1] + pad) / fx)
                        box[:, 1] = np.clip(box[:, 1], pad / fy, (im0.shape[0] + pad) / fy)

                        ###
                        box = np.array(box).reshape([-1, 2])

                        startidx = box.sum(axis=1).argmin()
                        box = np.roll(box, 4 - startidx, 0)

                        x = box[:, 0]
                        l_idx = x.argsort()
                        l_box = np.array([box[l_idx[0]], box[l_idx[1]]])
                        r_box = np.array([box[l_idx[2]], box[l_idx[3]]])
                        l_idx_1 = np.array(l_box[:, 1]).argsort()
                        lt = l_box[l_idx_1[0]]
                        lt[lt < 0] = 0
                        lb = l_box[l_idx_1[1]]
                        r_idx_1 = np.array(r_box[:, 1]).argsort()
                        rt = r_box[r_idx_1[0]]
                        rt[rt < 0] = 0
                        rb = r_box[r_idx_1[1]]

                        if pred_label == 0:
                            lt1 = lt
                            rt1 = rt
                            rb1 = rb
                            lb1 = lb
                        elif pred_label == 1:
                            lt1 = rt
                            rt1 = rb
                            rb1 = lb
                            lb1 = lt
                        elif pred_label == 2:
                            lt1 = rb
                            rt1 = lb
                            rb1 = lt
                            lb1 = rt
                        elif pred_label == 3:
                            lt1 = lb
                            rt1 = lt
                            rb1 = rt
                            lb1 = rb
                        else:
                            lt1 = lt
                            rt1 = rt
                            rb1 = rb
                            lb1 = lb

                        lb = [float(lb1[0]) * fx - pad, float(lb1[1]) * fy - pad]
                        lt = [float(lt1[0]) * fx - pad, float(lt1[1]) * fy - pad]
                        rt = [float(rt1[0]) * fx - pad, float(rt1[1]) * fy - pad]
                        rb = [float(rb1[0]) * fx - pad, float(rb1[1]) * fy - pad]
                        ###

                        # lb = [float(box[3, 0]) * fx - pad, float(box[3, 1]) * fy - pad]
                        # lt = [float(box[0, 0]) * fx - pad, float(box[0, 1]) * fy - pad]
                        # rt = [float(box[1, 0]) * fx - pad, float(box[1, 1]) * fy - pad]
                        # rb = [float(box[2, 0]) * fx - pad, float(box[2, 1]) * fy - pad]

                        result[filename].append({
                            "box": boundingRect,
                            "lb": lb,
                            "lt": lt,
                            "rt": rt,
                            "rb": rb,
                        })

    with open(os.path.join(save_dir, "result.txt"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)
