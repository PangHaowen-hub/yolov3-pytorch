import torch
from iou import bbox_iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):  # conf_thres置信度阈值
    # 将中心加宽高的形式转换为左上角右下角的形式
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # 左上角x
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # 左上角y
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # 右下角x
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # 右下角y
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]  # 生成一个列表，长度和prediction相同，为图片数目
    for image_i, image_pred in enumerate(prediction):  # 循环，有多少照片循环多少次
        # 利用置信度进行第一轮筛选
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()  # 保留所有置信度大于conf_thres的先验框
        image_pred = image_pred[conf_mask]  # image_pred为34*85
        if not image_pred.size(0):
            continue

        # 获得种类及其置信度
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # torch.max(input, dim, keepdim=False) dim=0求每列最大值，dim=1求每行最大值。class_conf为置信度数值，class_pred为第几类
        # keepdim=True表示保持原数据的shape
        # 获得的内容为(x1, y1, x2, y2, obj_conf置信度, class_conf类置信度, class_pred哪一类)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # 求这幅图中包含的种类
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # 获得某一类初步筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]

            # 按照置信度大小将所有这一类的框排序
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]

            # 进行非极大抑制
            max_detections = []
            while detections_class.size(0):
                # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:])  # 计算置信度最大的框与其他框的iou
                detections_class = detections_class[1:][ious < nms_thres]  # 若置信度最大的框与某框iou大于阈值，则舍弃某框
            # 堆叠
            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))

    return output
