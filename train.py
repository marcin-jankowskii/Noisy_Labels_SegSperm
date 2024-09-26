import cv2

from models.DualUNetPlusPlus import DualUNetPlusPlus
from models.QuadUNetPlusPlus import QuadUNetPlusPlus
from dataset.data import BatchMaker
from utils.metrics import calculate_iou, calculate_ap_for_segmentation
from utils.basic_augmentation import BasicAugmentation
from utils.class_specific_augmentation import ClassSpecificAugmentation
from utils.class_specific_augmentation_3masks import ClassSpecificAugmentation as ClassSpecificAugmentation2


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR
import datetime
import yaml
import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
import segmentation_models_pytorch as smp
import argparse


class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255],[0,255,255]]
resize_size = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--annotator', type=int, default=1, help='Annotator to use.')
    parser.add_argument('--model', type=str, default='smpUNet++', help='Model to use.')
    parser.add_argument('--augmentation', action='store_true', help='Use augmentation.')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='Loss function to use.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use.')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', help='Scheduler to use.')
    parser.add_argument('--place', type=str, default='lab', help='Place for config path.')
    parser.add_argument('--mode', type=str, default='two_task_training(4)', help='Mode of operation.')
    parser.add_argument('--aug_type', type=str, default='ClassSpecificAugmentation', help='Type of augmentation to use.')
    parser.add_argument('--k', type=int, default=5, help='K parameter.')
    return parser.parse_args()

def transform_mask(mask):
    if config.mode == 'two_task_training(4)':
        new_mask = np.zeros((resize_size, resize_size))
    else:
        new_mask = np.zeros((512, 512))
    for i in range(3):
        new_mask[mask[i] == 1] = i

    return new_mask

def transform_mask2(mask):
    if config.mode == 'two_task_training(4)':
        new_mask = np.zeros((resize_size, resize_size))
    else:
        new_mask = np.zeros((512, 512))
    for i in range(4):
        if i == 0:
            new_mask[mask[i] == 1] = i
        if i == 3:
            new_mask[mask[i] == 1] = 1
    return new_mask

def transform_batch(batch):
    if config.mode == 'two_task_training(4)':
        new_batch = np.zeros((batch.shape[0], resize_size, resize_size))
    else:
        new_batch = np.zeros((batch.shape[0], 512, 512))
    for i in range(batch.shape[0]):
        new_batch[i] = transform_mask(batch[i])

    return new_batch

def transform_batch2(batch):
    if config.mode == 'two_task_training(4)':
        new_batch = np.zeros((batch.shape[0], resize_size, resize_size))
    else:
        new_batch = np.zeros((batch.shape[0], 512, 512))

    for i in range(batch.shape[0]):
        new_batch[i] = transform_mask2(batch[i])

    return new_batch


def plot_sample(X, y, preds, ix=None,mode = 'train', number = '1'):
    """Function to plot the results"""
    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  
    colorsV2 = [[0, 0, 0], [0, 255, 0], [255, 0, 0],[0,255,255]]  
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 3,figsize=(20, 10))
    if X[ix].shape[-1] == 3:
        ax[0].imshow(X[ix])
    elif X[ix].shape[-1] == 6:

        first_image = X[ix][..., :3]
        second_image = X[ix][..., 3:]
        combined_image = np.concatenate((first_image, second_image), axis=1)
        ax[0].imshow(combined_image)
    #if has_mask:
        #ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Sperm Image')
    ax[0].set_axis_off()


    mask_to_display = y[ix]

    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask_rgb[mask_to_display == i] = color


    ax[1].imshow(mask_rgb)
    ax[1].set_title('Sperm Mask Image')
    ax[1].set_axis_off()


    mask_to_display = preds[ix]

    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask_rgb[mask_to_display == i] = color


    ax[2].imshow(mask_rgb)
    #if has_mask:
        #ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Sperm Image Predicted (class1 and class2)')
    ax[2].set_axis_off()




    if mode == 'train':
        wandb.log({"train/plot": wandb.Image(fig)})
    if mode == 'val':
        if number == '1':
            wandb.log({"val/plot": wandb.Image(fig)})
        else:
            wandb.log({"val/plot" + number: wandb.Image(fig)})

    plt.close()

def train(model, train_loader, optimizer,scheduler,loss_fn,augumentation,T_aug,epoch_number):
    model.train()
    total_loss = 0
    total_loss_oneclass = 0
    total_loss_multiclass = 0
    total_iou_multiclass = 0
    total_iou_oneclass = 0
    total_iou_multiclass2 = 0
    total_iou_oneclass2 = 0
    total_iou_multiclass3 = 0
    total_iou_oneclass3 = 0
    total_loss_oneclass_head2 = 0
    total_loss_multiclass_head2 = 0
    total_loss_oneclass_head3 = 0
    total_loss_multiclass_head3= 0
    total_loss_oneclass_head4 = 0
    total_loss_multiclass_head4 = 0

    for batch_idx, data in enumerate(train_loader):
        if len(data) == 2:
            inputs, ids = data
        elif len(data) == 7:
            inputs,intersections, unions,feelings,ids_y1,ids_y2,ids_y3 = data

        if T_aug == True:
            for i in range(inputs.shape[0]):
                if len(data) == 2:
                    inputs[i],ids[i] = augumentation(inputs[i], ids[i])
                elif len(data) == 7:
                    if config.annotator == 1:
                        if config.mode == 'two_task_training' or config.mode == 'two_task_training(4)':
                            inputs[i], ids_y1[i],ids_y2[i],ids_y3[i] = augumentation(inputs[i], ids_y1[i],ids_y2[i],ids_y3[i])
                        else:
                            inputs[i],ids_y1[i] = augumentation(inputs[i], ids_y1[i])
                    elif config.annotator == 2:
                        if config.mode == 'two_task_training' or config.mode == 'two_task_training(4)':
                            inputs[i], ids_y1[i],ids_y2[i],ids_y3[i] = augumentation(inputs[i], ids_y1[i],ids_y2[i],ids_y3[i])
                        else:
                            inputs[i],ids_y2[i] = augumentation(inputs[i], ids_y2[i])
                    elif config.annotator == 3:
                        if config.mode == 'two_task_training' or config.mode == 'two_task_training(4)':
                            inputs[i], ids_y1[i],ids_y2[i],ids_y3[i] = augumentation(inputs[i], ids_y1[i],ids_y2[i],ids_y3[i])
                        else:
                            inputs[i],ids_y3[i] = augumentation(inputs[i], ids_y3[i])
                    elif config.annotator == 4:
                        if config.mode == 'two_task_training' or config.mode == 'two_task_training(4)':
                            inputs[i], ids_y1[i],ids_y2[i],ids_y3[i] = augumentation(inputs[i], ids_y1[i],ids_y2[i],ids_y3[i])
                        else:
                            inputs[i],ids_y1[i],ids_y2[i] = augumentation(inputs[i], ids_y1[i],ids_y2[i])
                    elif config.annotator == 5:
                        if config.mode == 'two_task_training' or config.mode == 'two_task_training(4)':
                            inputs[i], ids_y1[i],ids_y2[i],ids_y3[i] = augumentation(inputs[i], ids_y1[i],ids_y2[i],ids_y3[i])
                        else:
                            inputs[i],unions[i] = augumentation(inputs[i], unions[i])

        inputs = inputs.to(device)
        if len(data) == 2:
            if config.loss == 'CrossEntropyLoss':
                ids = ids.type(torch.LongTensor)
            if config.loss == 'BCEWithLogitsLoss':
                ids = ids.type(torch.FloatTensor)
            ids = ids.to(device)
            idsBCE = ids[:, [0, -1], :, :]
            idsBCE = idsBCE.to(device)
        elif len(data) == 7:

            intersections = intersections.type(torch.LongTensor)
            intersections = intersections.to(device)
            feelings = feelings.type(torch.LongTensor)
            feelings = feelings.to(device)
            intersections1 = transform_batch(intersections.cpu())
            intersections2 = transform_batch2(intersections.cpu())
            intersections1 = torch.from_numpy(intersections1).type(torch.LongTensor).to(device)
            intersections2 = torch.from_numpy(intersections2).type(torch.LongTensor).to(device)

            ids_y1 =ids_y1.type(torch.LongTensor)
            ids_y1 = ids_y1.to(device)

            ids_y2 = ids_y2.type(torch.LongTensor)
            ids_y2 = ids_y2.to(device)

            ids_y3 = ids_y3.type(torch.LongTensor)
            ids_y3 = ids_y3.to(device)

            if config.annotator == 1:
                ids = ids_y1.to(device)
            elif config.annotator == 2:
                ids = ids_y2.to(device)
            elif config.annotator == 3:
                ids = ids_y3.to(device)
            elif config.annotator == 4:
                ids = intersections.to(device)
            elif config.annotator == 5:
                ids = unions.to(device)


            unions = unions.type(torch.LongTensor)
            unions = unions.to(device)
            unions1 = transform_batch(unions.cpu())
            unions2 = transform_batch2(unions.cpu())
            unions1 = torch.from_numpy(unions1).type(torch.LongTensor).to(device)
            unions2 = torch.from_numpy(unions2).type(torch.LongTensor).to(device)

            if config.mode == 'intersection_and_union':
                un_diff_inter = ids - intersections
            elif config.mode == 'cascade' or config.mode == 'cascade_con':
                un_diff_inter = ids - feelings
            if config.mode == 'intersection_and_union' or config.mode == 'cascade' or config.mode == 'cascade_con':
                un_diff_inter = un_diff_inter.type(torch.LongTensor)
                un_diff_inter = un_diff_inter.to(device)

                un_diff_inter1 = transform_batch(un_diff_inter.cpu())
                un_diff_inter2 = transform_batch2(un_diff_inter.cpu())

                un_diff_inter1 = torch.from_numpy(un_diff_inter1).type(torch.LongTensor).to(device)
                un_diff_inter2 = torch.from_numpy(un_diff_inter2).type(torch.LongTensor).to(device)

        if config.mode != 'two_task_training(4)':
            ids1 = transform_batch(ids.cpu())
            ids2 = transform_batch2(ids.cpu())


            ids1 = torch.from_numpy(ids1).type(torch.LongTensor).to(device)
            ids2 = torch.from_numpy(ids2).type(torch.LongTensor).to(device)


        optimizer.zero_grad()

        if config.mode == 'two_task_training':
            output_head1, output_head2 = model(inputs)
            output1 = output_head1[:, :3, :, :]
            output2 = output_head1[:, [0, -1], :, :]
            output3 = output_head2[:, :3, :, :]
            output4 = output_head2[:, [0, -1], :, :]
            predicted_output1 = torch.argmax(output1, dim=1)
            predicted_output2 = torch.argmax(output2, dim=1)

            same_class_mask1 = (ids1 == predicted_output1)
            same_class_mask2 = (ids2 == predicted_output2)

            ids1_head2 = torch.where(same_class_mask1, ids1 - predicted_output1, ids1)
            ids2_head2 = torch.where(same_class_mask2, ids2 - predicted_output2, ids2)
            mask1 = (predicted_output1 != ids1).float().unsqueeze(1)
            mask2 = (predicted_output2 != ids2).float().unsqueeze(1)

            output3_masked = output3 * mask1
            output4_masked = output4 * mask2

        elif config.mode == 'two_task_training(4)':

            ids1 = transform_batch(ids_y1.cpu())
            ids2 = transform_batch2(ids_y1.cpu())
            ids3 = transform_batch(ids_y2.cpu())
            ids4 = transform_batch2(ids_y2.cpu())
            ids5 = transform_batch(ids_y3.cpu())
            ids6 = transform_batch2(ids_y3.cpu())

            ids1 = torch.from_numpy(ids1).type(torch.LongTensor).to(device)
            ids2 = torch.from_numpy(ids2).type(torch.LongTensor).to(device)
            ids3 = torch.from_numpy(ids3).type(torch.LongTensor).to(device)
            ids4 = torch.from_numpy(ids4).type(torch.LongTensor).to(device)
            ids5 = torch.from_numpy(ids5).type(torch.LongTensor).to(device)
            ids6 = torch.from_numpy(ids6).type(torch.LongTensor).to(device)

            output_head1, output_head2, output_head3, output_head4 = model(inputs)
            output1 = output_head1[:, :3, :, :]
            output2 = output_head1[:, [0, -1], :, :]
            output3 = output_head2[:, :3, :, :]
            output4 = output_head2[:, [0, -1], :, :]
            output5 = output_head3[:, :3, :, :]
            output6 = output_head3[:, [0, -1], :, :]
            output7 = output_head4[:, :3, :, :]
            output8 = output_head4[:, [0, -1], :, :]

            predicted_output1 = torch.argmax(output1, dim=1)
            predicted_output2 = torch.argmax(output2, dim=1)

            same_class_mask1 = (ids1 == predicted_output1)
            same_class_mask2 = (ids2 == predicted_output2)
            same_class_mask3 = (ids3 == predicted_output1)
            same_class_mask4 = (ids4 == predicted_output2)
            same_class_mask5 = (ids5 == predicted_output1)
            same_class_mask6 = (ids6 == predicted_output2)

            ids1_head2 = torch.where(same_class_mask1, ids1 - predicted_output1, ids1)
            ids2_head2 = torch.where(same_class_mask2, ids2 - predicted_output2, ids2)
            ids3_head3 = torch.where(same_class_mask3, ids3 - predicted_output1, ids3)
            ids4_head3 = torch.where(same_class_mask4, ids4 - predicted_output2, ids4)
            ids5_head4 = torch.where(same_class_mask5, ids5 - predicted_output1, ids5)
            ids6_head4 = torch.where(same_class_mask6, ids6 - predicted_output2, ids6)

            mask1 = (predicted_output1 != ids1).float().unsqueeze(1)
            mask2 = (predicted_output2 != ids2).float().unsqueeze(1)
            mask3 = (predicted_output1 != ids3).float().unsqueeze(1)
            mask4 = (predicted_output2 != ids4).float().unsqueeze(1)
            mask5 = (predicted_output1 != ids5).float().unsqueeze(1)
            mask6 = (predicted_output2 != ids6).float().unsqueeze(1)

            output3_masked = output3 * mask1
            output4_masked = output4 * mask2
            output5_masked = output5 * mask3
            output6_masked = output6 * mask4
            output7_masked = output7 * mask5
            output8_masked = output8 * mask6

        else:

            output = model(inputs)
            output1 = output[:, :3, :, :]
            output2 = output[:, [0, -1], :, :]

            output3 = output1.clone()
            mask = feelings[:, :3] == 1
            output3[mask] = 0

            output4 = output2.clone()
            mask2 = feelings[:,[0,-1]] == 1
            output4[mask2] = 0

        weights1 = torch.tensor([0.2, 1.0, 0.5]).to(device)
        weights2 = torch.tensor([1.0, 1.0]).to(device)
        weights3 = torch.tensor([0.4, 1.0, 0.8]).to(device)
        loss_fn1 = nn.CrossEntropyLoss(weight=weights1)
        loss_fn2 = nn.CrossEntropyLoss(weight=weights2)
        loss_fn3 = nn.CrossEntropyLoss(weight=weights3)


        if config.mode == 'intersection_and_union' or config.mode == 'cascade' or config.mode == 'cascade_con':
            k = config.k
            l1 = loss_fn1(output1, ids1)  + k*loss_fn1(output3, un_diff_inter1)
            l2 = loss_fn2(output2, ids2) + k*loss_fn2(output4, un_diff_inter2)
            loss = l1 + l2
        elif config.mode == 'two_task_training':
            k = config.k
            l1 = loss_fn1(output1, intersections1)
            l2 = loss_fn2(output2, intersections2)

            l3 = loss_fn1(output3, ids1) + k*loss_fn3(output3_masked, ids1_head2)
            l4 = loss_fn2(output4, ids2) + k*loss_fn2(output4_masked, ids2_head2)
            loss1 = l1+l2
            loss2 = l3+l4
            loss = loss1 + loss2
        elif config.mode == 'two_task_training(4)':
            k = config.k
            l1 = loss_fn1(output1, intersections1)
            l2 = loss_fn2(output2, intersections2)

            l3 = loss_fn1(output3, ids1) + k*loss_fn3(output3_masked, ids1_head2)
            l4 = loss_fn2(output4, ids2) + k*loss_fn2(output4_masked, ids2_head2)
            l5 = loss_fn1(output5, ids3) + k*loss_fn3(output5_masked, ids3_head3)
            l6 = loss_fn2(output6, ids4) + k*loss_fn2(output6_masked, ids4_head3)
            l7 = loss_fn1(output7, ids5) + k*loss_fn3(output7_masked, ids5_head4)
            l8 = loss_fn2(output8, ids6) + k*loss_fn2(output8_masked, ids6_head4)

            loss1 = l1+l2
            loss2 = l3+l4
            loss3 = l5+l6
            loss4 = l7+l8

            loss = loss1 + loss2 + loss3 + loss4

        elif config.mode == 'oneclass':
            if config.loss == 'CrossEntropyLoss':
                loss = loss_fn2(output, ids2)
            elif config.loss == 'BCEWithLogitsLoss':
                loss = loss_fn(output, idsBCE)
        elif config.mode == 'multiclass' or config.mode == 'intersection':
            if config.loss == 'CrossEntropyLoss':
                l1 = loss_fn1(output1, ids1)
                l2 = loss_fn2(output2, ids2)
                loss = l1 + l2
            if config.loss == 'BCEWithLogitsLoss':
                loss = loss_fn(output, ids)

        if config.mode == 'oneclass':
            preds2 = torch.argmax(output, dim=1)
            mean_iou, IoUs = calculate_iou(ids2.cpu().numpy(), preds2.cpu().numpy(),2)
            iou_oneclass = 1 - mean_iou

            loss.backward()
            optimizer.step()

            total_iou_multiclass += 0
            total_iou_oneclass += iou_oneclass
            total_loss += loss.item()

        elif config.mode == 'multiclass' or config.mode == 'intersection_and_union' or config.mode == 'intersection' or config.mode == 'cascade' or config.mode == 'cascade_con' or config.mode == 'two_task_training' or config.mode == 'two_task_training(4)':

            if config.mode == 'two_task_training':
                preds1 = torch.argmax(output1 + output3, dim=1)
                preds2 = torch.argmax(output2 + output4, dim=1)

            elif config.mode == 'two_task_training(4)':
                preds1 = torch.argmax(output1 + output3, dim=1)
                preds2 = torch.argmax(output2 + output4, dim=1)
                preds3 = torch.argmax(output1 + output5, dim=1)
                preds4 = torch.argmax(output2 + output6, dim=1)
                preds5 = torch.argmax(output1 + output7, dim=1)
                preds6 = torch.argmax(output2 + output8, dim=1)
            else:
                preds1 = torch.argmax(output1, dim=1)
                preds2 = torch.argmax(output2, dim=1)


            mean_iou, IoUs = calculate_iou(ids1.cpu().numpy(), preds1.cpu().numpy(), 3)
            iou_multiclass = 1 - mean_iou
            mean_iou, IoUs = calculate_iou(ids2.cpu().numpy(), preds2.cpu().numpy(),2)
            iou_oneclass = 1 - mean_iou

            if config.mode == 'two_task_training(4)':
                mean_iou, IoUs = calculate_iou(ids3.cpu().numpy(), preds3.cpu().numpy(), 3)
                iou_multiclass2 = 1 - mean_iou
                mean_iou, IoUs = calculate_iou(ids4.cpu().numpy(), preds4.cpu().numpy(),2)
                iou_oneclass2 = 1 - mean_iou
                mean_iou, IoUs = calculate_iou(ids5.cpu().numpy(), preds5.cpu().numpy(), 3)
                iou_multiclass3 = 1 - mean_iou
                mean_iou, IoUs = calculate_iou(ids6.cpu().numpy(), preds6.cpu().numpy(),2)
                iou_oneclass3 = 1 - mean_iou
            else:
                iou_multiclass2 = 0
                iou_oneclass2 = 0
                iou_multiclass3 = 0
                iou_oneclass3 = 0


            loss.backward()
            optimizer.step()

            total_iou_multiclass += iou_multiclass
            total_iou_oneclass += iou_oneclass
            total_loss += loss.item()
            if config.loss == 'CrossEntropyLoss':
                total_loss_oneclass += l2.item()
                total_loss_multiclass += l1.item()
                if config.mode == 'two_task_training':
                    total_loss_oneclass_head2 += l4.item()
                    total_loss_multiclass_head2 += l3.item()
                elif config.mode == 'two_task_training(4)':
                    total_iou_multiclass2 += iou_multiclass2
                    total_iou_oneclass2 += iou_oneclass2
                    total_iou_multiclass3 += iou_multiclass3
                    total_iou_oneclass3 += iou_oneclass3

                    total_loss_oneclass_head2 += l4.item()
                    total_loss_multiclass_head2 += l3.item()
                    total_loss_oneclass_head3 += l6.item()
                    total_loss_multiclass_head3 += l5.item()
                    total_loss_oneclass_head4 += l8.item()
                    total_loss_multiclass_head4 += l7.item()
                else:
                    total_loss_oneclass_head2 = 0
                    total_loss_multiclass_head2 = 0
                    total_loss_oneclass_head3 = 0
                    total_loss_multiclass_head3 = 0
                    total_loss_oneclass_head4 = 0
                    total_loss_multiclass_head4 = 0



    if config.mode == 'two_task_training':
        avg_loss_head2 = total_loss_oneclass_head2 / len(train_loader)
        avg_loss_multiclass_head2 = total_loss_multiclass_head2 / len(train_loader)
    elif config.mode == 'two_task_training(4)':
        avg_loss_head2 = total_loss_oneclass_head2 / len(train_loader)
        avg_loss_multiclass_head2 = total_loss_multiclass_head2 / len(train_loader)
        avg_loss_head3 = total_loss_oneclass_head3 / len(train_loader)
        avg_loss_multiclass_head3 = total_loss_multiclass_head3 / len(train_loader)
        avg_loss_head4 = total_loss_oneclass_head4 / len(train_loader)
        avg_loss_multiclass_head4 = total_loss_multiclass_head4 / len(train_loader)

        avg_iou_multiclass2 = total_iou_multiclass2 / len(train_loader)
        avg_iou_oneclass2 = total_iou_oneclass2 / len(train_loader)
        avg_iou_multiclass3 = total_iou_multiclass3 / len(train_loader)
        avg_iou_oneclass3 = total_iou_oneclass3 / len(train_loader)

    avg_loss = total_loss / len(train_loader)
    avg_loss_oneclass = total_loss_oneclass / len(train_loader)
    avg_loss_multiclass = total_loss_multiclass / len(train_loader)
    avg_iou_multiclass = total_iou_multiclass / len(train_loader)
    avg_iou_oneclass = total_iou_oneclass / len(train_loader)


    if config.scheduler != 'ReduceLROnPlateau':
        scheduler.step()

    if config.mode == 'two_task_training':
        metrics = {"train/train_loss": avg_loss,
                "train/train_loss_oneclass": avg_loss_oneclass,
                "train/train_loss_multiclass": avg_loss_multiclass,
                "train/train_loss_oneclass_head2": avg_loss_head2,
                "train/train_loss_multiclass_head2": avg_loss_multiclass_head2,
               "train/train_iou_multiclass": avg_iou_multiclass,
               "train/train_iou_oneclass": avg_iou_oneclass,
               "train/lr": optimizer.param_groups[0]['lr'],
                       "train/epoch": epoch_number
                       }
    elif config.mode == 'two_task_training(4)':
        metrics = {"train/train_loss": avg_loss,
                "train/train_loss_oneclass": avg_loss_oneclass,
                "train/train_loss_multiclass": avg_loss_multiclass,
                "train/train_loss_oneclass_head2": avg_loss_head2,
                "train/train_loss_multiclass_head2": avg_loss_multiclass_head2,
                "train/train_loss_oneclass_head3": avg_loss_head3,
                "train/train_loss_multiclass_head3": avg_loss_multiclass_head3,
                "train/train_loss_oneclass_head4": avg_loss_head4,
                "train/train_loss_multiclass_head4": avg_loss_multiclass_head4,
               "train/train_iou_multiclass": avg_iou_multiclass,
               "train/train_iou_oneclass": avg_iou_oneclass,
                "train/train_iou_multiclass2": avg_iou_multiclass2,
                "train/train_iou_oneclass2": avg_iou_oneclass2,
                "train/train_iou_multiclass3": avg_iou_multiclass3,
                "train/train_iou_oneclass3": avg_iou_oneclass3,
               "train/lr": optimizer.param_groups[0]['lr'],
                       "train/epoch": epoch_number
                       }
    else:
        metrics = {"train/train_loss": avg_loss,
                    "train/train_loss_oneclass": avg_loss_oneclass,
                    "train/train_loss_multiclass": avg_loss_multiclass,
                   "train/train_iou_multiclass": avg_iou_multiclass,
                   "train/train_iou_oneclass": avg_iou_oneclass,
                   "train/lr": optimizer.param_groups[0]['lr'],
                           "train/epoch": epoch_number
                           }


    wandb.log(metrics)

    return avg_loss,avg_iou_multiclass,avg_iou_oneclass

def val(model, validation_loader, loss_fn,epoch_number,scheduler):
    model.eval()
    softmask_oneclass_list = []
    softmask_multiclass_list = []
    softmask_oneclass_list2 = []
    softmask_multiclass_list2 = []
    softmask_oneclass_list3 = []
    softmask_multiclass_list3 = []
    vids_list = []
    vids_list2 = []
    vids_list3 = []
    total_iou_multiclass = 0
    total_iou_oneclass = 0
    total_iou_multiclass2 = 0
    total_iou_oneclass2 = 0
    total_iou_multiclass3 = 0
    total_iou_oneclass3 = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):

            if len(data) == 2:
                vinputs, vids = data
                vids = vids.type(torch.FloatTensor)
                vids = vids.to(device)
            elif len(data) == 7:
                vinputs, vintersections, vunions, vfeelings, vids_y1, vids_y2, vids_y3 = data
                vintersections = vintersections.type(torch.FloatTensor)
                vids = vids_y1.to(device)
                vunions = vunions.type(torch.FloatTensor)

                if config.annotator == 1:
                    vids = vids_y1.to(device)
                elif config.annotator == 2:
                    vids = vids_y2.to(device)
                elif config.annotator == 3:
                    vids = vids_y3.to(device)
                elif config.annotator == 4:
                    vids = vintersections.to(device)
                elif config.annotator == 5:
                    vids = vunions.to(device)

            vinputs = vinputs.to(device)
            images = vinputs.detach().cpu().numpy().transpose(0, 2, 3, 1)

            if config.mode == 'oneclass':
                vids_list.append(vids.cpu())
                vids1 = transform_batch(vids.cpu())
                vids2 = transform_batch2(vids.cpu())



                vids1 = torch.from_numpy(vids1).type(torch.LongTensor).to(device)
                vids2 = torch.from_numpy(vids2).type(torch.LongTensor).to(device)

                vids_numpy = vids1.detach().cpu().numpy()
                vids_numpy2 = vids2.detach().cpu().numpy()

                voutputs = model(vinputs)

                vpreds2 = torch.argmax(voutputs, dim=1)
                vsofts2 = torch.softmax(voutputs, dim=1)
                vsofts2 = vsofts2.squeeze(0)
                softmask_oneclass_list.append(vsofts2.cpu())

                preds_out_oneclass = vpreds2.detach().cpu().numpy()
                preds_out_multiclass = np.zeros_like(preds_out_oneclass)

                total_iou_multiclass = 0

                mean_iou, IoUs = calculate_iou(vids2.cpu().numpy(), vpreds2.cpu().numpy(),2)

                viou = 1 - mean_iou
                total_iou_oneclass += viou
            else:
                if config.mode != 'two_task_training(4)':
                    vids_list.append(vids.cpu())
                    vids1 = transform_batch(vids.cpu())
                    vids2 = transform_batch2(vids.cpu())

                    vids1 = torch.from_numpy(vids1).type(torch.LongTensor).to(device)
                    vids2 = torch.from_numpy(vids2).type(torch.LongTensor).to(device)

                    vids_numpy = vids1.detach().cpu().numpy()
                    vids_numpy2 = vids2.detach().cpu().numpy()


                if config.mode == 'two_task_training':
                    voutputs_head1, voutputs_head2 = model(vinputs)
                    voutput1 = voutputs_head1[:, :3, :, :]
                    voutput2 = voutputs_head1[:, [0, -1], :, :]
                    voutput3 = voutputs_head2[:, :3, :, :]
                    voutput4 = voutputs_head2[:, [0, -1], :, :]
                elif config.mode == 'two_task_training(4)':

                    vids_list.append(vids_y1.cpu())
                    vids_list2.append(vids_y2.cpu())
                    vids_list3.append(vids_y3.cpu())

                    vids1 = transform_batch(vids_y1.cpu())
                    vids2 = transform_batch2(vids_y1.cpu())
                    vids3 = transform_batch(vids_y2.cpu())
                    vids4 = transform_batch2(vids_y2.cpu())
                    vids5 = transform_batch(vids_y3.cpu())
                    vids6 = transform_batch2(vids_y3.cpu())

                    vids1 = torch.from_numpy(vids1).type(torch.LongTensor).to(device)
                    vids2 = torch.from_numpy(vids2).type(torch.LongTensor).to(device)
                    vids3 = torch.from_numpy(vids3).type(torch.LongTensor).to(device)
                    vids4 = torch.from_numpy(vids4).type(torch.LongTensor).to(device)
                    vids5 = torch.from_numpy(vids5).type(torch.LongTensor).to(device)
                    vids6 = torch.from_numpy(vids6).type(torch.LongTensor).to(device)

                    vids_numpy = vids1.detach().cpu().numpy()
                    vids_numpy2 = vids2.detach().cpu().numpy()



                    voutputs_head1, voutputs_head2, voutputs_head3, voutputs_head4 = model(vinputs)
                    voutput1 = voutputs_head1[:, :3, :, :]
                    voutput2 = voutputs_head1[:, [0, -1], :, :]
                    voutput3 = voutputs_head2[:, :3, :, :]
                    voutput4 = voutputs_head2[:, [0, -1], :, :]
                    voutput5 = voutputs_head3[:, :3, :, :]
                    voutput6 = voutputs_head3[:, [0, -1], :, :]
                    voutput7 = voutputs_head4[:, :3, :, :]
                    voutput8 = voutputs_head4[:, [0, -1], :, :]

                else:

                    voutputs = model(vinputs)
                    voutput1 = voutputs[:, :3, :, :]
                    voutput2 = voutputs[:, [0, -1], :, :]

                if config.mode == 'two_task_training':
                    vpreds1 = torch.argmax(voutput1 + voutput3, dim=1)
                    vpreds2 = torch.argmax(voutput2 + voutput4, dim=1)
                    vsofts1 = torch.softmax(voutput1 + voutput3, dim=1)
                    vsofts2 = torch.softmax(voutput2 + voutput4, dim=1)
                elif config.mode == 'two_task_training(4)':
                    vpreds1 = torch.argmax(voutput1 + voutput3, dim=1)
                    vpreds2 = torch.argmax(voutput2 + voutput4, dim=1)
                    vpreds3 = torch.argmax(voutput1 + voutput5, dim=1)
                    vpreds4 = torch.argmax(voutput2 + voutput6, dim=1)
                    vpreds5 = torch.argmax(voutput1 + voutput7, dim=1)
                    vpreds6 = torch.argmax(voutput2 + voutput8, dim=1)

                    vsofts1 = torch.softmax(voutput1 + voutput3, dim=1)
                    vsofts2 = torch.softmax(voutput2 + voutput4, dim=1)
                    vsofts3 = torch.softmax(voutput1 + voutput5, dim=1)
                    vsofts4 = torch.softmax(voutput2 + voutput6, dim=1)
                    vsofts5 = torch.softmax(voutput1 + voutput7, dim=1)
                    vsofts6 = torch.softmax(voutput2 + voutput8, dim=1)

                    vsofts3 = vsofts3.squeeze(0)
                    vsofts4 = vsofts4.squeeze(0)
                    vsofts5 = vsofts5.squeeze(0)
                    vsofts6 = vsofts6.squeeze(0)

                else:
                    vpreds1 = torch.argmax(voutput1, dim=1)
                    vpreds2 = torch.argmax(voutput2, dim=1)
                    vsofts1 = torch.softmax(voutput1, dim=1)
                    vsofts2 = torch.softmax(voutput2, dim=1)


                vsofts1 = vsofts1.squeeze(0)
                vsofts2 = vsofts2.squeeze(0)
                softmask_multiclass_list.append(vsofts1.cpu())
                softmask_oneclass_list.append(vsofts2.cpu())

                if config.mode == 'two_task_training(4)':
                    softmask_multiclass_list2.append(vsofts3.cpu())
                    softmask_oneclass_list2.append(vsofts4.cpu())
                    softmask_multiclass_list3.append(vsofts5.cpu())
                    softmask_oneclass_list3.append(vsofts6.cpu())


                preds_out_multiclass = vpreds1.detach().cpu().numpy()
                preds_out_oneclass = vpreds2.detach().cpu().numpy()


                mean_iou, IoUs = calculate_iou(vids1.cpu().numpy(), vpreds1.cpu().numpy(),3)
                viou = 1 - mean_iou
                total_iou_multiclass += viou


                mean_iou, IoUs = calculate_iou(vids2.cpu().numpy(), vpreds2.cpu().numpy(),2)
                viou = 1 - mean_iou
                total_iou_oneclass += viou

                if config.mode == 'two_task_training(4)':
                    mean_iou, IoUs = calculate_iou(vids3.cpu().numpy(), vpreds3.cpu().numpy(),3)
                    viou = 1 - mean_iou
                    total_iou_multiclass2 += viou

                    mean_iou, IoUs = calculate_iou(vids4.cpu().numpy(), vpreds4.cpu().numpy(),2)
                    viou = 1 - mean_iou
                    total_iou_oneclass2 += viou

                    mean_iou, IoUs = calculate_iou(vids5.cpu().numpy(), vpreds5.cpu().numpy(),3)
                    viou = 1 - mean_iou
                    total_iou_multiclass3 += viou

                    mean_iou, IoUs = calculate_iou(vids6.cpu().numpy(), vpreds6.cpu().numpy(),2)
                    viou = 1 - mean_iou
                    total_iou_oneclass3 += viou


    vids_list = np.concatenate(vids_list, axis=0)
    ids1 = transform_batch(vids_list)
    ids2 = transform_batch2(vids_list)

    if config.mode == 'two_task_training(4)':
        vids_list2 = np.concatenate(vids_list2, axis=0)
        ids3 = transform_batch(vids_list2)
        ids4 = transform_batch2(vids_list2)
        vids_list3 = np.concatenate(vids_list3, axis=0)
        ids5 = transform_batch(vids_list3)
        ids6 = transform_batch2(vids_list3)

        softmask_onceclass2 = [mask for batch in softmask_oneclass_list2 for mask in batch]
        softmask_onceclass2_np = np.array(softmask_onceclass2)
        softmask_onceclass2_np = softmask_onceclass2_np.transpose(0, 2, 3, 1)
        ap_score_oneclass2 = calculate_ap_for_segmentation(softmask_onceclass2_np[:, :, :, 1], ids4)
        ap_score_head2 = 0
        ap_score_tail2 = 0

        softmask_oneclass3 = [mask for batch in softmask_oneclass_list3 for mask in batch]
        softmask_oneclass3_np = np.array(softmask_oneclass3)
        softmask_oneclass3_np = softmask_oneclass3_np.transpose(0, 2, 3, 1)
        ap_score_oneclass3 = calculate_ap_for_segmentation(softmask_oneclass3_np[:, :, :, 1], ids6)
        ap_score_head3 = 0
        ap_score_tail3 = 0


    softmasks_oneclass = [mask for batch in softmask_oneclass_list for mask in batch]
    softmasks_oneclass_np = np.array(softmasks_oneclass)
    softmasks_oneclass_np = softmasks_oneclass_np.transpose(0, 2, 3, 1)
    ap_score_oneclass = calculate_ap_for_segmentation(softmasks_oneclass_np[:, :, :, 1], ids2)
    ap_score_head = 0
    ap_score_tail = 0

    if config.mode == 'multiclass' or config.mode == 'intersection_and_union' or config.mode == 'intersection' or config.mode == 'cascade' or config.mode == 'cascade_con' or config.mode == 'two_task_training' or config.mode == 'two_task_training(4)':
        softmasks_multiclass = [mask for batch in softmask_multiclass_list for mask in batch]
        softmasks_multiclass_np = np.array(softmasks_multiclass)
        softmasks_multiclass_np = softmasks_multiclass_np.transpose(0, 2, 3, 1)
        ap_score_head = calculate_ap_for_segmentation(softmasks_multiclass_np[:, :, :, 2], vids_list[:,2,:,:])
        ap_score_tail = calculate_ap_for_segmentation(softmasks_multiclass_np[:, :, :, 1], vids_list[:,1,:,:])
        if config.mode == 'two_task_training(4)':
            softmask_multiclass2 = [mask for batch in softmask_multiclass_list2 for mask in batch]
            softmask_multiclass2_np = np.array(softmask_multiclass2)
            softmask_multiclass2_np = softmask_multiclass2_np.transpose(0, 2, 3, 1)
            ap_score_head2 = calculate_ap_for_segmentation(softmask_multiclass2_np[:, :, :, 2], vids_list2[:,2,:,:])
            ap_score_tail2 = calculate_ap_for_segmentation(softmask_multiclass2_np[:, :, :, 1], vids_list2[:,1,:,:])

            softmask_multiclass3 = [mask for batch in softmask_multiclass_list3 for mask in batch]
            softmask_multiclass3_np = np.array(softmask_multiclass3)
            softmask_multiclass3_np = softmask_multiclass3_np.transpose(0, 2, 3, 1)
            ap_score_head3 = calculate_ap_for_segmentation(softmask_multiclass3_np[:, :, :, 2], vids_list3[:,2,:,:])
            ap_score_tail3 = calculate_ap_for_segmentation(softmask_multiclass3_np[:, :, :, 1], vids_list3[:,1,:,:])
        else:
            ap_score_oneclass2 = 0
            ap_score_head2 = 0
            ap_score_tail2 = 0
            ap_score_oneclass3 = 0
            ap_score_head3 = 0
            ap_score_tail3 = 0


    avg_iou_multiclass = total_iou_multiclass / len(validation_loader)
    avg_iou_oneclass = total_iou_oneclass / len(validation_loader)

    if config.mode == 'two_task_training(4)':
        avg_iou_multiclass2 = total_iou_multiclass2 / len(validation_loader)
        avg_iou_oneclass2 = total_iou_oneclass2 / len(validation_loader)
        avg_iou_multiclass3 = total_iou_multiclass3 / len(validation_loader)
        avg_iou_oneclass3 = total_iou_oneclass3 / len(validation_loader)


    if config.scheduler == 'ReduceLROnPlateau':
        scheduler.step(avg_iou_multiclass)

    if config.mode == 'two_task_training(4)':
        val_metrics = { "val/val_iou_multiclass": avg_iou_multiclass,
                    "val/val_iou_oneclass": avg_iou_oneclass,
                    "val/val_iou_multiclass2": avg_iou_multiclass2,
                    "val/val_iou_oneclass2": avg_iou_oneclass2,
                    "val/val_iou_multiclass3": avg_iou_multiclass3,
                    "val/val_iou_oneclass3": avg_iou_oneclass3,
                    "val/val_ap_oneclass": ap_score_oneclass,
                    "val/val_ap_head": ap_score_head,
                    "val/val_ap_tail": ap_score_tail,
                    "val/val_ap_oneclass2": ap_score_oneclass2,
                    "val/val_ap_head2": ap_score_head2,
                    "val/val_ap_tail2": ap_score_tail2,
                    "val/val_ap_oneclass3": ap_score_oneclass3,
                    "val/val_ap_head3": ap_score_head3,
                    "val/val_ap_tail3": ap_score_tail3,
                    "val/epoch": epoch_number
                       }
    else:

        val_metrics = { "val/val_iou_multiclass": avg_iou_multiclass,
                        "val/val_iou_oneclass": avg_iou_oneclass,
                        "val/val_ap_oneclass": ap_score_oneclass,
                        "val/val_ap_head": ap_score_head,
                        "val/val_ap_tail": ap_score_tail,
                        "val/epoch": epoch_number
                           }
    wandb.log(val_metrics)
    return avg_iou_multiclass,avg_iou_oneclass,images,vids_numpy,vids_numpy2,preds_out_multiclass,preds_out_oneclass,ap_score_oneclass,ap_score_head,ap_score_tail,

def main(model, train_loader, validation_loader, optimizer,scheduler,loss_fn, epochs,augumentation,T_aug,name):

    best_iou = 1000000
    best_iou_multiclass = 1000000
    best_iou_opt_oneclass = 0
    best_ap_oneclass = 0
    best_ap_head = 0
    best_ap_tail = 0

    for epoch in range(epochs):
        epoch_number = epoch +1
        train_loss,train_iou_multiclass,train_iou_oneclass = train(model, train_loader, optimizer,scheduler, loss_fn,augumentation,T_aug,epoch_number)
        validation_iou_multiclass,validation_iou_oneclass,vimages,vlbls_multiclass,vlbls_oneclass,vpreds_multiclass,vpreds_oneclass,ap_score_oneclass,ap_score_head,ap_score_tail = val(model, validation_loader, loss_fn,epoch_number,scheduler)
        plot_sample(vimages,vlbls_multiclass,vpreds_multiclass, ix=0,mode = 'val',number = '1')
        plot_sample(vimages, vlbls_oneclass, vpreds_oneclass, ix=0, mode='val',number = '2')

        print(f'Epoch {epoch_number}, Train Loss: {train_loss}, Train Iou Multiclass: {train_iou_multiclass}, Train Iou Oneclass: {train_iou_oneclass}, Validation Iou Multiclass: {validation_iou_multiclass}, Validation Iou Oneclass: {validation_iou_oneclass}')
        print(f'Validation AP Oneclass: {ap_score_oneclass}',f'Validation AP Head: {ap_score_head}',f'Validation AP Tail: {ap_score_tail}')

        if validation_iou_oneclass < best_iou:
            best_iou = validation_iou_oneclass
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_iou_oneclass'
            torch.save(model.state_dict(), model_path)
            print('Model saved (iou_oneclass)')
        if ap_score_oneclass > best_ap_oneclass:
            best_ap_oneclass = ap_score_oneclass
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_ap_oneclass'
            torch.save(model.state_dict(), model_path)
            print("Model saved (ap_oneclass)")
        if ap_score_head > best_ap_head:
            best_ap_head = ap_score_head
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_ap_head'
            torch.save(model.state_dict(), model_path)
            print("Model saved (ap_head)")
        if ap_score_tail > best_ap_tail:
            best_ap_tail = ap_score_tail
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_ap_tail'
            torch.save(model.state_dict(), model_path)
            print("Model saved (ap_tail)")
        if validation_iou_multiclass < best_iou_multiclass:
            best_iou_multiclass = validation_iou_multiclass
            model_path = yaml_config['save_model_path'] +'/'+ name + '_best_model_iou_multiclass'
            torch.save(model.state_dict(), model_path)
            print('Model saved (iou_multiclass)')
        if epoch_number == epochs:
            model_path = yaml_config['save_model_path'] +'/'+ name + '_last_model'
            torch.save(model.state_dict(), model_path)
            print('Model saved')



class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255],[0,255,255]]  


timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

args = parse_args()

wandb.init(project="Noisy_label", entity="noisy_label",
            config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "annotator": args.annotator, #annotator 4 - intersection, 5 - union
            "model": args.model,
            "augmentation": args.augmentation,
            "loss": args.loss,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "place": args.place,
            "mode": args.mode,
            "aug_type": args.aug_type,
            "k": args.k
            })

config = wandb.config


if config.mode == 'oneclass':
    num_classes = 2
else:
    num_classes = 4

if config.mode == 'cascade_con':
    size_input = 7
elif config.mode == 'cascade':
    size_input = 4
else:
    size_input = 3


path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/marcin/463C6E583C6E42D3/Projects/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            }

model_dict = {'smpUNet': smp.Unet(in_channels = size_input, classes=num_classes),
              'smpUNet++': smp.UnetPlusPlus(in_channels = size_input, classes=num_classes,encoder_name="resnet18",encoder_weights=None),
              'MAnet': smp.MAnet(in_channels = size_input, classes=num_classes,encoder_name="resnet18",encoder_weights=None),
              'DeepLabV3+': smp.DeepLabV3Plus(in_channels = size_input, classes=num_classes,encoder_name="resnet18",encoder_weights=None),
              'DualUNetPlusPlus': DualUNetPlusPlus(size_input,num_classes),
              'QuadUNetPlusPlus': QuadUNetPlusPlus(size_input,num_classes)
}

mode_dict = {'normal': 'intersection_inference',
             'intersection': 'intersection_inference',
             'intersection_and_union': 'intersection_and_union_inference',
             'feeling_lucky': 'feeling_lucky',
             'union': 'union',
             "oneclass": 'mixed',
             "multiclass": 'mixed',
             "cascade": 'cascade',
             "cascade_con": 'cascade_con',
             "two_task_training": 'two_task_training',
             "two_task_training(4)": 'two_task_training(4)'
}

name = (f'5_Annotator_{config.annotator}_k={config.k}_Model_{config.model}_Augmentation_{config.augmentation}_Mode{config.mode}_Optimizer_{config.optimizer}_Scheduler_{config.scheduler}_Epochs_{config.epochs}_Batch_Size_{config.batch_size}_Start_lr_{config.lr}_Loss_{config.loss}_Timestamp_{timestamp}')

if config.mode == 'intersection_and_union' or config.mode == 'cascade' or config.mode == 'cascade_con':
    description = f'Two_segment'
else:
    description = f'One_segment_loss'
wandb.run.name = description + name
save_name = description + name




with open(path_dict[config.place], 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

batch_maker = BatchMaker(config_path=path_dict[config.place], batch_size=config.batch_size,mode ='train',segment = mode_dict[config.mode],annotator= config.annotator)
train_loader = batch_maker.train_loader
val_loader = batch_maker.test_loader


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU dostępne:", gpu_name )
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")

if config.mode == 'two_task_training':
    model = model_dict["DualUNetPlusPlus"]
elif config.mode == 'two_task_training(4)':
    model = model_dict["QuadUNetPlusPlus"]
else:
    model = model_dict[config.model]
model.to(device)

optimizer_dict = {'Adam': optim.Adam(model.parameters(), lr=config.lr),
                  'SGD': optim.SGD(model.parameters(), lr=config.lr),
                  'RMSprop': optim.RMSprop(model.parameters(), lr=config.lr)
                  }

if config.mode == "multiclass" or config.mode == "intersection_and_union" or config.mode == "intersection" or config.mode == "cascade" or config.mode == "cascade_con" or config.mode == "two_task_training" or config.mode == "two_task_training(4)":
    weights = torch.ones([num_classes,512,512])
    weights[0] = 1
    weights[1] = 10 #7
    weights[2] = 7  #2
    weights[3] = 8  #4
if config.mode =="oneclass":
    weights = torch.ones([num_classes,512,512])
    weights[0] = 1
    weights[1] = 1
    weights = weights.to(device)

loss_dict = {'CrossEntropyLoss': nn.CrossEntropyLoss(),
             'CrossEntropyLossWeight': nn.CrossEntropyLoss(),
             'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(pos_weight=weights),
             'BCE': nn.BCELoss()}

loss_fn = loss_dict[wandb.config.loss]

optimizer = optimizer_dict[config.optimizer]

scheduler_dict = {'CosineAnnealingLR': CosineAnnealingLR(optimizer, T_max=config.epochs),
                  'ReduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min'),
                  "MultiStepLR": MultiStepLR(optimizer, milestones=[30, 80], gamma=0.3),
                  'None': None}

scheduler = scheduler_dict[config.scheduler]
wandb.watch(model, log="all")
if config.aug_type == 'BasicAugmentation':
    aug = BasicAugmentation()
elif config.aug_type == 'ClassSpecificAugmentation':
    aug = ClassSpecificAugmentation()
    if config.mode == 'intersection_and_union' or config.annotator == 4 or config.mode == 'two_task_training' or config.mode == 'two_task_training(4)':
        aug = ClassSpecificAugmentation2()
t_aug = config.augmentation
print(config.aug_type)

if __name__ == "__main__":
    main(model, train_loader, val_loader, optimizer, scheduler, loss_fn, config.epochs, aug, t_aug, save_name)


