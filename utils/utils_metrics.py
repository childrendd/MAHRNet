import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算系数
    # --------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


# 设标签宽W，长H
def fast_hist(a, b, n):
    # a是标签，形状(H×W,)；b是预测结果，形状(H×W,)
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    #hist = fast_hist(label.flatten(), pred.flatten(), num_classes)



def per_class_iu(hist):   #计算iou   不需要输入图片，根据混淆矩阵计算
     return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)



def per_class_PA_Recall(hist):  #计算召回率
     return np.diag(hist) / np.maximum(hist.sum(1), 1)



def per_class_Precision(hist):  #计算precision
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):   #计算accuracy
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def per_class_Dice(hist):
    epsilon = 1e-5
    per_class_dice = np.zeros(hist.shape[0])
    for i in range(hist.shape[0]):
        # 计算Dice分数
        tp = hist[i, i]
        fp = np.sum(hist[:, i]) - tp
        fn = np.sum(hist[i, :]) - tp
        #per_class_dice[i] = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        per_class_dice[i] = (2 * tp ) / (2 * tp + fp + fn )
    return per_class_dice


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None): #需要输入图片
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        # 在每个图像处理完成后打印Dice指标
        Dice = per_class_Dice(hist)
        print('{:d} / {:d}: mIoU-{:0.2f}%; mPA-{:0.2f}%; Precision-{:0.2f}%; Dice-{:0.2f}%'.format(
            ind,
            len(gt_imgs),
            100 * np.nanmean(per_class_iu(hist)),
            100 * np.nanmean(per_class_PA_Recall(hist)),
            100 * np.nanmean(per_class_Precision(hist)),
            100 * np.nanmean(Dice)
        ))

        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIoU-{:0.2f}%; mPA-{:0.2f}%; Precision-{:0.2f}%; Dice-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * np.nanmean(per_class_Precision(hist)),
                100 * np.nanmean(Dice)
            )
            )

    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    Dice = per_class_Dice(hist)
    Accuracy = per_Accuracy(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2))
                  + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
                round(Precision[ind_class] * 100, 2)) + '; Dice-' + str(round(Dice[ind_class] * 100, 2)))

    print('===> IoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; Recall: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Precision: ' + str(round(np.nanmean(Precision) * 100, 2)) +
          '; Dice: ' + str(round(np.nanmean(Dice) * 100, 2)))

    return np.array(hist, int), IoUs, PA_Recall, Precision, Dice


def adjust_axes(r, t, fig, axes):     #用于调整图表坐标轴的函数。
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, 
                   output_path, tick_font_size=12, plt_show=True): #
    fig = plt.gcf()
    axes = plt.gca()                                                           
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)    
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, Dice, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, "IoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union",
                   os.path.join(miou_out_path, "mIoUcasediceadamb4we0hr'.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIOUcasediceadamb4we0hr'.png"))

    # draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy",
    #                os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    # print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "Recall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall",
                   os.path.join(miou_out_path, "Recall-casediceadamb4we0hr'.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall-casediceadamb4we0hr'.png"))

    draw_plot_func(Precision, name_classes, "Precision = {0:.2f}%".format(np.nanmean(Precision) * 100), "Precision",
                   os.path.join(miou_out_path, "Precision-casediceadamb4we0hr"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision-casediceb4we0hr'.png"))

    draw_plot_func(Dice, name_classes, "Dice = {0:.2f}%".format(np.nanmean(Dice) * 100), "Dice",
                   os.path.join(miou_out_path, "Dice-casediceadamwe0hr'.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Dice out to " + os.path.join(miou_out_path, "Dice-casediceadamb4we0hr'.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix-casediceadamb4we0hr'.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix-casediceadamb4we0hr'.csv"))
