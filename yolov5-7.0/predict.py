# 代码示例
# python predict.py [src_image_dir] [results]

import os
import sys
import glob
import json
import time
from pathlib import Path

import cv2

import paddle
import math

import cv2
import numpy as np
import glob
import os


def resize_to_test(img, sz=(640, 480)):
    imw, imh = sz
    return cv2.resize(np.float32(img), (imw, imh), cv2.INTER_CUBIC)


def decode_image(img, resize=False, sz=(640, 480)):
    imw, imh = sz
    img = np.squeeze(np.minimum(np.maximum(img, 0.0), 1.0))
    if resize:
        img = resize_to_test(img, sz=(imw, imh))
    img = np.uint8(img * 255.0)
    if len(img.shape) == 2:
        return np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
    else:
        return img


def det_lines(image):
    h, w = image.shape
    horizontal = image.copy()
    vertical = image.copy()

    for i in range(w):
        start_index = -1
        count = 0
        for j in range(h):
            if start_index == -1:
                if horizontal[j, i] == 255:
                    start_index = j
                    count = 1
            else:
                if horizontal[j, i] == 255:
                    count = count + 1
                else:
                    if count < 20:
                        for k in range(count):
                            horizontal[start_index + k, i] = 0

                    start_index = -1
                    count = 0

    for i in range(h):
        start_index = -1
        count = 0
        for j in range(w):
            if start_index == -1:
                if vertical[i, j] == 255:
                    start_index = j
                    count = 1
            else:
                if vertical[i, j] == 255:
                    count = count + 1
                else:
                    if count < 20:
                        for k in range(count):
                            vertical[i, start_index + k] = 0

                    start_index = -1
                    count = 0

    col_sum = horizontal.sum(axis=0)  # 纵向投影
    row_sum = vertical.sum(axis=1)  # 横向投影

    col_point = []
    start_index = -1
    count = 0
    for i in range(len(col_sum)):
        if start_index == -1:
            if col_sum[i] > 30:
                start_index = i
                count = 1
        else:
            if col_sum[i] > 30:
                count = count + 1
            else:
                cur_point = start_index + count / 2
                if len(col_point) > 0:
                    last_point = col_point[len(col_point) - 1]
                    if cur_point - last_point < 6:  # 与上一个点靠得太近, 取它们的中点更新上一个点
                        col_point[len(col_point) - 1] = (cur_point + last_point) / 2
                    else:
                        col_point.append(cur_point)
                else:
                    col_point.append(start_index + count / 2)
                start_index = -1
                count = 0

    row_point = []
    start_index = -1
    count = 0
    for i in range(len(row_sum)):
        if start_index == -1:
            if row_sum[i] > 30:
                start_index = i
                count = 1
        else:
            if row_sum[i] > 30:
                count = count + 1
            else:
                cur_point = start_index + count / 2
                if len(row_point) > 0:
                    last_point = row_point[len(row_point) - 1]
                    if cur_point - last_point < 6:  # 与上一个点靠得太近, 取它们的中点更新上一个点
                        row_point[len(row_point) - 1] = (cur_point + last_point) / 2
                    else:
                        row_point.append(cur_point)
                else:
                    row_point.append(cur_point)
                start_index = -1
                count = 0

    box = [col_point[0], row_point[0], col_point[len(col_point) - 1], row_point[len(row_point) - 1]]
    col_tables = []
    for i in range(len(col_point) - 1):
        x0, y0 = col_point[i], box[1]
        x1, y1 = col_point[i + 1], box[3]
        col_tables.append((x0, y0, x1, y1))

    row_tables = []
    for i in range(len(row_point) - 1):
        x0, y0 = box[0], row_point[i]
        x1, y1 = box[2], row_point[i + 1]
        row_tables.append((x0, y0, x1, y1))

    return box, col_tables, row_tables


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

    t = time.time()
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


def process_test(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    test_h = 480
    test_w = 480

    # paddle.disable_static()
    # params = paddle.load(r'best_paddle_model/model.pdparams')
    # model = DetectionModel()
    # model.set_dict(params, use_structured_name=False)
    # model.eval()

    ##########
    import paddle.inference as pdi
    w = 'best_paddle_model'
    if not Path(w).is_file():  # if not *.pdmodel
        w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
    weights = Path(w).with_suffix('.pdiparams')
    config = pdi.Config(str(w), str(weights))
    if True:
        config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
    predictor = pdi.create_predictor(config)
    input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    output_names = predictor.get_output_names()

    result = {}

    for image_path in image_paths:
        # image_path = 'dataset/table/images/border_0_0AUFQKYOR2PMM5IQJGBK.jpg'
        filename = os.path.split(image_path)[1]
        print(image_path)

        # do something
        im0 = cv2.imread(image_path)

        # im = letterbox(im0, (480, 480), stride=32, auto=True)[0]  # padded resize
        im = im0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)[None]  # contiguous
        im = im.astype(np.float32) / 255.0

        input_handle.copy_from_cpu(im)
        predictor.run()
        pred = [predictor.get_output_handle(x).copy_to_cpu() for x in output_names]

        # pred = model(img).numpy()
        pred = non_max_suppression_np(pred[0], 0.5, 0.45, classes=[0])

        pred_t = np.load('pred_np.npy')
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes([test_h, test_w], det[:, :4], im0.shape).round()

                for i, det0 in enumerate(det):
                    x0, y0, x1, y1 = det0[:4]
                    conf = det0[4]
                    cls = det0[5]

                    img_show = im0.copy()
                    cv2.rectangle(img_show, (int(x0), int(y0)), (int(x1), int(y1)), (128, 33, 1), 2)
                    cv2.imshow('img_show', img_show)
                    cv2.waitKey(0)

    with open(os.path.join(save_dir, "result.txt"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(result))


def decode_image(img, resize=False, sz=(640, 480)):
    imw, imh = sz
    img = np.squeeze(np.minimum(np.maximum(img, 0.0), 1.0))
    if resize:
        img = resize_to_test(img, sz=(imw, imh))
    img = np.uint8(img * 255.0)
    if len(img.shape) == 2:
        return np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
    else:
        return img


def find_box(mask):
    col_sum = mask.sum(axis=0)  # 纵向投影
    row_sum = mask.sum(axis=1)  # 横向投影

    for i in range(len(col_sum)):
        if col_sum[i] > 30:
            pass


def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1)
    bw = cv2.bitwise_not(bw)

    ###########################################
    horizontal = bw.copy()
    vertical = bw.copy()
    img = image.copy()

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    horizontal = cv2.dilate(horizontal, (1, 1), iterations=5)
    horizontal = cv2.erode(horizontal, (1, 1), iterations=5)

    # HoughlinesP function to detect horizontal lines
    hor_lines = cv2.HoughLinesP(horizontal, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=3)
    if hor_lines is None:
        return None, None
    temp_line = []
    for line in hor_lines:
        for x1, y1, x2, y2 in line:
            temp_line.append([x1, y1, x2, y2])

    # Sorting the list of detected lines by Y1
    hor_lines = sorted(temp_line, key=lambda x: x[1])

    # Selection of best lines from all the horizontal lines detected ##
    lasty1 = -111111
    lines_x1 = []
    lines_x2 = []
    hor = []
    i = 0
    for x1, y1, x2, y2 in hor_lines:
        if y1 >= lasty1 and y1 <= lasty1 + 10:
            lines_x1.append(x1)
            lines_x2.append(x2)
        else:
            if (i != 0 and len(lines_x1) is not 0):
                hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
            lasty1 = y1
            lines_x1 = []
            lines_x2 = []
            lines_x1.append(x1)
            lines_x2.append(x2)
            i += 1
    hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
    #####################################################################

    # [vertical lines]
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.dilate(vertical, (1, 1), iterations=8)
    vertical = cv2.erode(vertical, (1, 1), iterations=7)

    ver_lines = cv2.HoughLinesP(vertical, 1, np.pi / 180, 50, np.array([]), 20, 2)
    if ver_lines is None:
        return None, None
    temp_line = []
    for line in ver_lines:
        for x1, y1, x2, y2 in line:
            temp_line.append([x1, y1, x2, y2])

    # Sorting the list of detected lines by X1
    ver_lines = sorted(temp_line, key=lambda x: x[0])

    ## Selection of best lines from all the vertical lines detected ##
    lastx1 = -111111
    lines_y1 = []
    lines_y2 = []
    ver = []
    count = 0
    lasty1 = -11111
    lasty2 = -11111
    for x1, y1, x2, y2 in ver_lines:
        if x1 >= lastx1 and x1 <= lastx1 + 15 and not (
                ((min(y1, y2) < min(lasty1, lasty2) - 20 or min(y1, y2) < min(lasty1, lasty2) + 20)) and (
                (max(y1, y2) < max(lasty1, lasty2) - 20 or max(y1, y2) < max(lasty1, lasty2) + 20))):
            lines_y1.append(y1)
            lines_y2.append(y2)

        else:
            if count != 0 and len(lines_y1) is not 0:
                ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])
            lastx1 = x1
            lines_y1 = []
            lines_y2 = []
            lines_y1.append(y1)
            lines_y2.append(y2)
            count += 1
            lasty1 = -11111
            lasty2 = -11111
    ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])

    return hor, ver


def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))

    test_h = 480
    test_w = 480

    import paddle.inference as pdi

    config = pdi.Config('best_paddle_model/inference_model/model.pdmodel',
                        'best_paddle_model/inference_model/model.pdiparams')
    config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
    predictor = pdi.create_predictor(config)
    input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
    output_names = predictor.get_output_names()

    config_gan = pdi.Config('pd_model/inference_model/model.pdmodel',
                            'pd_model/inference_model/model.pdiparams')
    config_gan.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
    predictor_gan = pdi.create_predictor(config_gan)
    input_handle_gan = predictor_gan.get_input_handle(predictor_gan.get_input_names()[0])
    output_names_gan = predictor_gan.get_output_names()

    result = {}
    label_names = ["spanning_cell", "row", "column", "table"]

    for image_path in image_paths:
        filename = os.path.split(image_path)[1]

        # do something
        im0 = cv2.imread(image_path)

        pad = 32
        h, w, c = im0.shape
        new_h, new_w = h + 2 * pad, w + 2 * pad
        img_pad = np.zeros(shape=(new_h, new_w, 3), dtype=np.uint8)
        img_pad[pad:pad + h, pad:pad + w] = im0

        src = img_pad.copy()
        im = cv2.resize(np.float32(img_pad), (test_w, test_h), cv2.INTER_CUBIC)

        im = im.astype(np.float32) / 255.0

        # gan
        input_handle_gan.copy_from_cpu(im[None])
        predictor_gan.run()
        pred_gan = [predictor_gan.get_output_handle(x).copy_to_cpu() for x in output_names_gan]
        out_gan = decode_image(pred_gan)

        # _, mask = cv2.threshold(out_gan[:, :, 0], 128, 1, cv2.THRESH_BINARY)
        # hor, ver = line_detection(mask)

        # yolov5
        im[out_gan[:, :, 0] > 127] = (0, 0, 1)

        # cv2.imshow('out_gan', out_gan)
        # cv2.imshow('yolo_in', (im*255).astype(np.uint8))

        im2 = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im2 = np.ascontiguousarray(im2)

        input_handle.copy_from_cpu(im2[None])
        predictor.run()
        pred = [predictor.get_output_handle(x).copy_to_cpu() for x in output_names]

        pred = non_max_suppression_np(pred[0], 0.25, 0.45, classes=[0, 1, 2])

        fx = src.shape[1] / test_w
        fy = src.shape[0] / test_h

        if filename not in result:
            result[filename] = []

        box = [1000, 1000, 0, 0]
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes([test_h, test_w], det[:, :4], im.shape).round()

                for i, det0 in enumerate(det):
                    x0, y0, x1, y1 = det0[:4]
                    conf = det0[4]
                    cls = det0[5]

                    if int(cls) == 0 and conf < 0.8:
                        continue

                    obj_box = [x0 * fx - pad, y0 * fy - pad, x1 * fx - pad, y1 * fy - pad]
                    result[filename].append({
                        "box": obj_box,
                        "label": label_names[int(cls)]
                    })

                    # show_img = im0.copy()
                    # cv2.rectangle(show_img, (int(obj_box[0]), int(obj_box[1])), (int(obj_box[2]), int(obj_box[3])), (234, 33, 54), 3)
                    # cv2.imshow('df', show_img)
                    # cv2.waitKey(0)

                    if int(cls) == 1:
                        box[0] = min(x0, box[0])
                        box[1] = min(y0, box[1])
                        box[2] = max(x1, box[2])
                        box[3] = max(y1, box[3])

        result[filename].append({
            "box": [box[0] * fx - pad, box[1] * fy - pad, box[2] * fx - pad, box[3] * fy - pad],
            "label": label_names[3]
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
