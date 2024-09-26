import torch
import yaml
import glob
import cv2
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from skimage.transform import resize
import torch.nn.functional as F
import random
from natsort import natsorted
from scipy.stats import mode
from numba import njit

path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            }

path_config = {"place": "lab"
               }



def rgb_to_class_id(mask_rgb, class_colors):
        mask_id = np.zeros((*mask_rgb.shape[:2], len(class_colors)), dtype=np.float32)
        for class_id, color in enumerate(class_colors):
            idx = class_id
            if class_id == 3:
                idx = 1
                mask = (mask_rgb == color).all(axis=2).astype(np.float32)
                mask_id[:, :, -1] = np.logical_or(mask_id[:, :, -1], mask)
            else:
                mask = (mask_rgb == color).all(axis=2).astype(np.float32)
                mask_id[:, :, idx] = mask

            if idx == 1 or idx == 2:
                mask_id[:, :, -1] = np.logical_or(mask_id[:, :, -1], mask)

        return mask_id


def mean_voting(mask1, mask2, mask3):
    mean_mask = (mask1 + mask2 + mask3) / 3.0
    result_mask = (mean_mask > 0.5).astype(int)
    return result_mask

def majority_voting(mask1, mask2, mask3):
    vote_mask = mask1 + mask2 + mask3
    result_mask = (vote_mask > 1.5).astype(int)
    return result_mask

def median_voting(mask1, mask2, mask3):
    stacked_masks = np.stack([mask1, mask2, mask3], axis=0)
    result_mask = np.median(stacked_masks, axis=0).astype(int)
    return result_mask


@njit
def mode_aggregation(mask1, mask2, mask3):
    n, num_classes, height, width = mask1.shape
    result_mask = np.zeros((n, num_classes, height, width), dtype=np.int32)

    for i in range(n):
        for j in range(num_classes):
            for k in range(height):
                for l in range(width):
                    pixel_values = np.array([mask1[i, j, k, l], mask2[i, j, k, l], mask3[i, j, k, l]], dtype=np.int32)
                    result_mask[i, j, k, l] = np.bincount(pixel_values).argmax()

    return result_mask

@njit
def feeling_lucky(mask1, mask2, mask3):
    n, num_classes, height, width = mask1.shape
    result_mask = np.zeros((n, num_classes, height, width), dtype=mask1.dtype)

    for i in range(n):
        for j in range(num_classes):
            for k in range(height):
                for l in range(width):
                    random_choice = np.random.randint(3)
                    if random_choice == 0:
                        result_mask[i, j, k, l] = mask1[i, j, k, l]
                    elif random_choice == 1:
                        result_mask[i, j, k, l] = mask2[i, j, k, l]
                    else:
                        result_mask[i, j, k, l] = mask3[i, j, k, l]

    return result_mask


class ProcessData:
    def __init__(self, config_path=path_dict[path_config['place']], mode = 'full',annotator = 1):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
            self.mode = mode
            self.annotator = annotator


    def process_dataset(self, dataset_name):
        dataset_path = self.config['dataset_path']
        dataset_path = dataset_path + dataset_name
        print(dataset_path)



        if self.annotator == 1:
            name = '/GT1_'
        elif self.annotator == 2:
            name = '/GT2_'
        elif self.annotator ==3:
            name = '/GT3_'
        if self.mode == 'full':
            segment = 'full/'
        elif self.mode == 'head':
            segment = 'head/'
        elif self.mode == 'tail':
            segment = 'tail/'
        elif self.mode == 'mixed':
            segment = 'mixed/'    
        
        class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255],[0,255,255]] 

        images = natsorted(glob.glob(f"{dataset_path}/images/*"))

        if self.mode == 'intersection_and_union' or self.mode == 'intersection' or self.mode == 'intersection_and_union_inference' or self.mode == 'intersection_inference' or self.mode == 'feeling_lucky' or self.mode == 'union' or self.mode == 'cascade' or self.mode == 'cascade_con' or self.mode == 'two_task_training' or self.mode == 'all_agregation' or self.mode == 'two_task_training(4)':
            gt_path1 = dataset_path + '/GT1_' + 'mixed/'
            gt_path2 = dataset_path + '/GT2_' + 'mixed/'
            gt_path3 = dataset_path + '/GT3_' + 'mixed/'

            if dataset_name == '/train':
                pred_one_segment_path = '/media/marcin/463C6E583C6E42D3/Projects/noisy_labels_data/pred_one_segment_train/'
            elif dataset_name == '/test':
                pred_one_segment_path = '/media/marcin/463C6E583C6E42D3/Projects/noisy_labels_data/pred_one_segment_test/'

            gt1_path_for_pred = pred_one_segment_path + 'GT1/'
            gt2_path_for_pred = pred_one_segment_path + 'GT2/'
            gt3_path_for_pred = pred_one_segment_path + 'GT3/'


            masks = natsorted(glob.glob(f"{gt_path1}*.png"))
            masks2 = natsorted(glob.glob(f"{gt_path2}*.png"))
            masks3 = natsorted(glob.glob(f"{gt_path3}*.png"))

            images_pred = natsorted(glob.glob(f"{pred_one_segment_path}*.png"))
            mask_sorted = natsorted(glob.glob(f"{gt1_path_for_pred}*.png"))
            mask2_sorted = natsorted(glob.glob(f"{gt2_path_for_pred}*.png"))
            mask3_sorted = natsorted(glob.glob(f"{gt3_path_for_pred}*.png"))


            if self.mode == 'two_task_training(4)':

                resize_size = 512
                X = np.zeros((len(masks), resize_size, resize_size, 3),dtype=np.float32)
                intersections = np.zeros((len(masks), resize_size, resize_size, 4), dtype=np.float32)
                unions = np.zeros((len(masks), resize_size, resize_size, 4), dtype=np.float32)
                feelings = np.zeros((len(masks), resize_size, resize_size, 4), dtype=np.float32)
                y1 = np.zeros((len(masks), resize_size, resize_size, 4), dtype=np.float32)
                y2 = np.zeros((len(masks), resize_size, resize_size, 4), dtype=np.float32)
                y3 = np.zeros((len(masks), resize_size, resize_size, 4), dtype=np.float32)
            else:

                X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
                intersections = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
                unions = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
                feelings = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
                y1 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
                y2 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
                y3 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)


            if self.mode == 'cascade':
                X = np.zeros((len(images_pred), self.config['image_height'], self.config['image_width'], 4), dtype=np.float32)
                images = images_pred

            if self.mode == 'cascade_con':
                X = np.zeros((len(images_pred), self.config['image_height'], self.config['image_width'], 7), dtype=np.float32)
                for n,(img, img_pred, mimg, mimg2, mimg3) in enumerate(zip(images,images_pred,masks,masks2,masks3)):
                    img = cv2.imread(img)
                    x_img = img.astype(np.float32)
                    x_img = resize(x_img, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)

                    min_val = np.min(x_img)
                    max_val = np.max(x_img)
                    x_img = (x_img - min_val) / (max_val - min_val)

                    img_pred = cv2.imread(img_pred)
                    x_img_pred = img_pred.astype(np.float32)
                    x_img_pred = resize(x_img_pred, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                    feeling_id = rgb_to_class_id(x_img_pred, class_colors)

                    min_val = np.min(x_img_pred)
                    max_val = np.max(x_img_pred)
                    x_img_pred = (x_img_pred - min_val) / (max_val - min_val)

                    x_img = np.concatenate((x_img,feeling_id),axis=2)

                    mask = cv2.imread(mimg)
                    mask = mask.astype(np.float32)
                    mask = resize(mask, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                    mask_id = rgb_to_class_id(mask, class_colors)

                    mask2 = cv2.imread(mimg2)
                    mask2 = mask2.astype(np.float32)
                    mask2 = resize(mask2, (512, 512, 3), mode='constant', preserve_range=True)
                    mask2_id = rgb_to_class_id(mask2, class_colors)

                    mask3 = cv2.imread(mimg3)
                    mask3 = mask3.astype(np.float32)
                    mask3 = resize(mask3, (512, 512, 3), mode='constant', preserve_range=True)
                    mask3_id = rgb_to_class_id(mask3, class_colors)


                    intersection = cv2.bitwise_and(mask, mask2)
                    intersection = cv2.bitwise_and(intersection, mask3)

                    intersection_id = cv2.bitwise_and(mask_id, mask2_id)
                    intersection_id = cv2.bitwise_and(intersection_id, mask3_id)

                    union_id = cv2.bitwise_or(mask_id, mask2_id)
                    union_id = cv2.bitwise_or(union_id, mask3_id)


                    X[n] = x_img
                    intersections[n] = intersection_id
                    unions[n] = union_id
                    feelings[n] = feeling_id
                    y1[n] = mask_id
                    y2[n] = mask2_id
                    y3[n] = mask3_id


                if self.mode == 'intersection_and_union_inference' or self.mode == 'intersection_inference' or self.mode == 'cascade' or self.mode == 'cascade_con' or self.mode == 'two_task_training' or self.mode == 'two_task_training(4)':
                    return X, intersections, unions,feelings,y1, y2, y3
                elif self.mode == 'intersection_and_union' or self.mode == 'intersection':
                    return X, y2, intersections
                elif self.mode == 'feeling_lucky':
                    return X, feelings, unions
                elif self.mode == 'union':
                    return X, unions,intersections
            else:

                for n, (img, mimg,mimg2,mimg3) in enumerate(zip(images, masks, masks2, masks3)):
                    img = cv2.imread(img)
                    x_img = img.astype(np.float32)
                    if self.mode == 'two_task_training(4)':
                        x_img = resize(x_img, (resize_size, resize_size, 3), mode='constant', preserve_range=True)
                    else:
                        x_img = resize(x_img, (512, 512, 3), mode='constant', preserve_range=True)
                    if self.mode == 'cascade':
                        feeling_id = rgb_to_class_id(x_img, class_colors)
                        x_img = rgb_to_class_id(x_img, class_colors)

                    min_val = np.min(x_img)
                    max_val = np.max(x_img)
                    x_img = (x_img - min_val) / (max_val - min_val)

                    mask = cv2.imread(mimg)
                    mask = mask.astype(np.float32)
                    if self.mode == 'two_task_training(4)':
                        mask = resize(mask, (resize_size, resize_size, 3), mode='constant', preserve_range=True)
                    else:
                        mask = resize(mask, (512, 512, 3), mode='constant', preserve_range=True)
                    mask_id = rgb_to_class_id(mask, class_colors)

                    mask2 = cv2.imread(mimg2)
                    mask2 = mask2.astype(np.float32)
                    if self.mode == 'two_task_training(4)':
                        mask2 = resize(mask2, (resize_size, resize_size, 3), mode='constant', preserve_range=True)
                    else:
                        mask2 = resize(mask2, (512, 512, 3), mode='constant', preserve_range=True)
                    mask2_id = rgb_to_class_id(mask2, class_colors)

                    mask3 = cv2.imread(mimg3)
                    mask3 = mask3.astype(np.float32)
                    if self.mode == 'two_task_training(4)':
                        mask3 = resize(mask3, (resize_size, resize_size, 3), mode='constant', preserve_range=True)
                    else:
                        mask3 = resize(mask3, (512, 512, 3), mode='constant', preserve_range=True)
                    mask3_id = rgb_to_class_id(mask3, class_colors)


                    intersection = cv2.bitwise_and(mask, mask2)
                    intersection = cv2.bitwise_and(intersection, mask3)

                    intersection_id = cv2.bitwise_and(mask_id, mask2_id)
                    intersection_id = cv2.bitwise_and(intersection_id, mask3_id)

                    union_id = cv2.bitwise_or(mask_id, mask2_id)
                    union_id = cv2.bitwise_or(union_id, mask3_id)


                    if self.mode != 'cascade':
                        feeling_id = np.zeros_like(mask_id)

                    min_val = np.min(intersection)
                    max_val = np.max(intersection)

                    if (max_val - min_val) > 0:
                        intersection = (intersection - min_val) / (max_val - min_val)
                    else:
                        intersection = intersection / 255

                    if self.mode == 'intersection_and_union' or self.mode == 'intersection_and_union_inference':
                        X[n] = intersection
                    else:
                        X[n] = x_img
                    intersections[n] = intersection_id
                    unions[n] = union_id
                    feelings[n] = feeling_id
                    y1[n] = mask_id
                    y2[n] = mask2_id
                    y3[n] = mask3_id


                if self.mode == 'intersection_and_union_inference'  or self.mode == 'cascade':
                    return X, intersections, unions,feelings,y1, y2, y3
                elif self.mode == 'intersection_and_union' or self.mode == 'intersection':
                    return X, y2, intersections
                elif self.mode == 'feeling_lucky':
                    return X, feelings, unions
                elif self.mode == 'union':
                    return X, unions,intersections
                elif self.mode == 'all_agregation' or self.mode == 'intersection_inference' or self.mode == "two_task_training" or self.mode == 'two_task_training(4)':
                    return X, y1, y2, y3


        elif self.mode == 'both':
            gt_path1 = dataset_path + '/GT1_' + 'mixed/'
            gt_path2 = dataset_path + '/GT2_' + 'mixed/'
            masks = sorted(glob.glob(f"{gt_path1}*.png"))
            masks2 = sorted(glob.glob(f"{gt_path2}*.png"))

            X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
            y1 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
            y2 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)


            for n, (img, mimg,mimg2) in enumerate(zip(images, masks, masks2)):

                img = cv2.imread(img)
                x_img = img.astype(np.float32)
                x_img = resize(x_img, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)

                min_val = np.min(x_img)
                max_val = np.max(x_img)
                x_img = (x_img - min_val) / (max_val - min_val)

                mask = cv2.imread(mimg)
                mask = mask.astype(np.float32)
                mask = resize(mask, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                mask_id = rgb_to_class_id(mask, class_colors)
              
                mask2 = cv2.imread(mimg2)
                mask2 = mask2.astype(np.float32)
                mask2 = resize(mask2, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                mask2_id = rgb_to_class_id(mask2, class_colors)

                X[n] = x_img
                y1[n] = mask_id
                y2[n] = mask2_id

            return X, y1,y2
        


        else:
            gt_path = dataset_path + name + segment
            masks = sorted(glob.glob(f"{gt_path}*.png"))

            X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
            y = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
            



            for n, (img, mimg) in enumerate(zip(images, masks)):

                img = cv2.imread(img)
                x_img = img.astype(np.float32)
                x_img = resize(x_img, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)

                min_val = np.min(x_img)
                max_val = np.max(x_img)
                x_img = (x_img - min_val) / (max_val - min_val)

                mask = cv2.imread(mimg)
                mask = mask.astype(np.float32)
                mask = resize(mask, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                mask_id = rgb_to_class_id(mask, class_colors)


                X[n] = x_img
                y[n] = mask_id

            return X, y

class BatchMaker:
    def __init__(self, config_path=path_dict[path_config['place']], batch_size=6, mode = 'all',segment = 'full' ,annotator = 1):
        
    
        self.process_data = ProcessData(config_path=config_path,mode = segment,annotator = annotator)
        self.batch_size = batch_size
        if segment == 'intersection_and_union' or segment == 'intersection' or segment == 'both' or segment == 'feeling_lucky' or segment == 'union':
            if mode == 'all':
                x_train, int_train,un_train = self.process_data.process_dataset('/train')
                x_val, int_val,un_val = self.process_data.process_dataset('/test_small')
                x_test, int_test,un_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader2(x_train, int_train,un_train,shuffle=False)
                self.val_loader = self.create_loader2(x_val, int_val,un_val, shuffle=False)
                self.test_loader = self.create_loader2(x_test, int_test,un_test ,shuffle=False)
            elif mode == 'train':
                x_train, int_train,un_train = self.process_data.process_dataset('/train')
                x_val, int_val,un_val = self.process_data.process_dataset('/test_small')
                self.train_loader = self.create_loader2(x_train, int_train,un_train,shuffle=True)
                self.val_loader = self.create_loader2(x_val, int_val,un_val, shuffle=True)
            elif mode == 'test':
                x_test, int_test,un_test = self.process_data.process_dataset('/train')
                self.test_loader = self.create_loader2(x_test, int_test,un_test ,shuffle=False)


        elif segment == 'intersection_and_union_inference' or segment == 'cascade' or segment == 'cascade_con':
            if mode == 'all':
                x_train, int_train,un_train,fl_train,y1_train,y2_train,y3_train = self.process_data.process_dataset('/train')
                x_val, int_val,un_val,fl_val,y1_val,y2_val,y3_val = self.process_data.process_dataset('/test_small')
                x_test, int_test,un_test,fl_test,y1_test,y2_test,y3_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader3(x_train, int_train,un_train,fl_train,y1_train,y2_train,y3_train,shuffle=False)
                self.val_loader = self.create_loader3(x_val, int_val,un_val,fl_val,y1_val,y2_val,y3_val,shuffle=False)
                self.test_loader = self.create_loader3(x_test, int_test,un_test,fl_test,y1_test,y2_test,y3_test,shuffle=False)
            elif mode == 'train':
                x_train, int_train,un_train,fl_train,y1_train,y2_train,y3_train = self.process_data.process_dataset('/train')
                x_test, int_test,un_test,fl_test,y1_test,y2_test,y3_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader3(x_train, int_train,un_train,fl_train,y1_train,y2_train,y3_train,shuffle=True)
                self.test_loader = self.create_loader3(x_test, int_test,un_test,fl_test,y1_test,y2_test,y3_test,shuffle=True)
            elif mode == 'test':
                x_test, int_test,un_test,fl_test,y1_test,y2_test,y3_test = self.process_data.process_dataset('/test')
                self.test_loader = self.create_loader3(x_test, int_test,un_test,fl_test,y1_test,y2_test,y3_test,shuffle=False)

        elif segment == 'all_agregation':
                x_train, y1_train, y2_train, y3_train = self.process_data.process_dataset('/train')
                self.train_loader = self.create_loader_aggregation(x_train, y1_train, y2_train, y3_train, shuffle=False)


        elif segment == 'intersection_inference' or segment == 'two_task_training' or segment == 'two_task_training(4)':
            if mode == 'train':
                x_train, y1_train, y2_train, y3_train = self.process_data.process_dataset('/train')
                x_test, y1_test, y2_test, y3_test = self.process_data.process_dataset('/test')

                self.train_loader = self.create_loader4(x_train, y1_train, y2_train, y3_train, shuffle=True)
                self.test_loader = self.create_loader4(x_test, y1_test, y2_test, y3_test, shuffle=True)
            elif mode == 'test':
                x_test, y1_test, y2_test, y3_test = self.process_data.process_dataset('/test')
                self.test_loader = self.create_loader4(x_test, y1_test, y2_test, y3_test, shuffle=False)




        else:
            if mode == 'all':
                x_train, y_train = self.process_data.process_dataset('/train')
                x_val, y_val = self.process_data.process_dataset('/test_small')
                x_test, y_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader(x_train, y_train,shuffle=False)
                self.val_loader = self.create_loader(x_val, y_val, shuffle=False)
                self.test_loader = self.create_loader(x_test, y_test ,shuffle=False)
            elif mode == 'train':
                x_train, y_train = self.process_data.process_dataset('/train')
                x_test, y_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader(x_train, y_train, shuffle=True)
                self.test_loader = self.create_loader(x_test, y_test, shuffle=True)
            elif mode == 'test':
                x_test, y_test = self.process_data.process_dataset('/test')
                self.test_loader = self.create_loader(x_test, y_test, shuffle=False)
        

    def create_loader(self, x, y, shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        y = np.transpose(y, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y).type(torch.float64)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    

    def create_loader2(self, x, intersection,union,shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        intersection = np.transpose(intersection, (0, 3, 1, 2))
        union = np.transpose(union, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(x)
        intersection_tensor = torch.from_numpy(intersection).type(torch.float64)
        union_tensor = torch.from_numpy(union).type(torch.float64)
        dataset = TensorDataset(x_tensor, intersection_tensor,union_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def create_loader3(self,x,intersection,union,feeling,y1,y2,y3,shuffle):
        x = np.transpose(x,(0,3,1,2))
        intersection = np.transpose(intersection, (0, 3, 1, 2))
        union = np.transpose(union, (0, 3, 1, 2))
        feeling = np.transpose(feeling, (0, 3, 1, 2))
        y1 = np.transpose(y1, (0, 3, 1, 2))
        y2 = np.transpose(y2, (0, 3, 1, 2))
        y3 = np.transpose(y3, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(x)
        intersection_tensor = torch.from_numpy(intersection).type(torch.float64)
        union_tensor = torch.from_numpy(union).type(torch.float64)
        feeling_tensor = torch.from_numpy(feeling).type(torch.float64)
        y1_tensor = torch.from_numpy(y1).type(torch.float64)
        y2_tensor = torch.from_numpy(y2).type(torch.float64)
        y3_tensor = torch.from_numpy(y3).type(torch.float64)
        dataset = TensorDataset(x_tensor, intersection_tensor,union_tensor,feeling_tensor,y1_tensor,y2_tensor,y3_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def create_loader_aggregation(self, x, y1, y2, y3, shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        y1 = np.transpose(y1, (0, 3, 1, 2))
        y2 = np.transpose(y2, (0, 3, 1, 2))
        y3 = np.transpose(y3, (0, 3, 1, 2))

        mean_result = mean_voting(y1, y2, y3)
        print("mean loaded")
        majority_result = majority_voting(y1, y2, y3)
        print("majority loaded")
        median_result = median_voting(y1, y2, y3)
        print("median loaded")
        mode_result = mode_aggregation(y1, y2, y3)
        print("mode loaded")
        feeling_result = feeling_lucky(y1, y2, y3)
        print("feeling loaded")
        intersection_result = np.logical_and.reduce([y1, y2, y3]).astype(np.float32)
        print("intersection loaded")
        union_result = np.logical_or.reduce([y1, y2, y3]).astype(np.float32)
        print("union loaded")

        mean_tensor = torch.from_numpy(mean_result).type(torch.float64)
        majority_tensor = torch.from_numpy(majority_result).type(torch.float64)
        median_tensor = torch.from_numpy(median_result).type(torch.float64)
        mode_tensor = torch.from_numpy(mode_result).type(torch.float64)
        feeling_tensor = torch.from_numpy(feeling_result).type(torch.float64)
        intersection_tensor = torch.from_numpy(intersection_result).type(torch.float64)
        union_tensor = torch.from_numpy(union_result).type(torch.float64)
        y1_tensor = torch.from_numpy(y1).type(torch.float64)
        y2_tensor = torch.from_numpy(y2).type(torch.float64)
        y3_tensor = torch.from_numpy(y3).type(torch.float64)
        x_tensor = torch.from_numpy(x)

        dataset = TensorDataset(x_tensor, y1_tensor, y2_tensor, y3_tensor, mean_tensor, majority_tensor, median_tensor,
                                mode_tensor, feeling_tensor, intersection_tensor, union_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)


    def create_loader4(self,x,y1,y2,y3,shuffle):
        x = np.transpose(x,(0,3,1,2))
        y1 = np.transpose(y1, (0, 3, 1, 2))
        y2 = np.transpose(y2, (0, 3, 1, 2))
        y3 = np.transpose(y3, (0, 3, 1, 2))


        feeling_result = feeling_lucky(y1, y2, y3)
        print("feeling loaded")
        intersection_result = np.logical_and.reduce([y1, y2, y3]).astype(np.float32)
        print("intersection loaded")
        union_result = np.logical_or.reduce([y1, y2, y3]).astype(np.float32)
        print("union loaded")

        feeling_tensor = torch.from_numpy(feeling_result).type(torch.float64)
        intersection_tensor = torch.from_numpy(intersection_result).type(torch.float64)
        union_tensor = torch.from_numpy(union_result).type(torch.float64)
        x_tensor = torch.from_numpy(x)
        y1_tensor = torch.from_numpy(y1).type(torch.float64)
        y2_tensor = torch.from_numpy(y2).type(torch.float64)
        y3_tensor = torch.from_numpy(y3).type(torch.float64)

        dataset = TensorDataset(x_tensor, intersection_tensor, union_tensor, feeling_tensor, y1_tensor, y2_tensor,
                                y3_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
