import kornia as K
import kornia.augmentation as Ka
import torch
import torch.nn as nn
import random
import numpy as np
import cv2
import math


class ClassSpecificAugmentation(nn.Module):
    def __init__(self):
        super(ClassSpecificAugmentation, self).__init__()
        self.k1 = Ka.ColorJitter(brightness=0.2, contrast=0.3, p=0.5)
        self.k2 = Ka.RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False)
        self.k3 = Ka.RandomVerticalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False)
        self.k6 = Ka.RandomRotation(45.0, same_on_batch=False, align_corners=True, p=0.5, keepdim=False,
                                    resample='nearest')
        self.k5 = Ka.RandomGaussianBlur((3, 9), (0.1, 2.0), p=0.5)

        self.k4 = Ka.RandomResizedCrop((512, 512), scale=(0.67, 0.67), ratio=(0.75, 1.333), same_on_batch=False,
                                       resample='bilinear', p=0.5, align_corners=True)
        self.resize = Ka.Resize((512, 512))

    def forward(self, img: torch.Tensor, mask: torch.Tensor, mask2: torch.Tensor, mask3: torch.Tensor) -> torch.Tensor:

        mask_out = mask
        mask_out2 = mask2
        mask_out3 = mask3
        img_out = img
        img_out, mask_out, mask_out2, mask_out3 = self.copy_paste_sperm(img_out, mask_out, mask_out2, mask_out3)
        img_out = self.add_lines_along_tail(img_out, mask_out)
        img_out = self.change_brightness(img_out, mask_out)
        img_out = self.add_circles(img_out, mask_out)
        img_out = self.add_blur_along_tail(img_out, mask_out)
        img_out = self.k1(img_out)
        img_out = self.k2(img_out)
        img_out = self.k3(img_out)
        img_out = self.k4(img_out)
        img_out = self.k5(img_out)
        img_out = self.k6(img_out)
        img_out = self.resize(img_out)
        mask_out = self.k2(mask_out, self.k2._params)
        mask_out = self.k3(mask_out, self.k3._params)
        mask_out = self.k4(mask_out, self.k4._params)
        mask_out = self.k6(mask_out, self.k6._params)
        mask_out = self.resize(mask_out)
        mask_out2 = self.k2(mask_out2, self.k2._params)
        mask_out2 = self.k3(mask_out2, self.k3._params)
        mask_out2 = self.k4(mask_out2, self.k4._params)
        mask_out2 = self.k6(mask_out2, self.k6._params)
        mask_out2 = self.resize(mask_out2)
        mask_out3 = self.k2(mask_out3, self.k2._params)
        mask_out3 = self.k3(mask_out3, self.k3._params)
        mask_out3 = self.k4(mask_out3, self.k4._params)
        mask_out3 = self.k6(mask_out3, self.k6._params)
        mask_out3 = self.resize(mask_out3)


        return img_out, mask_out, mask_out2, mask_out3

    def find_contours(self, mask: torch.Tensor):
        mask_np = mask.numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def calculate_endpoint(self, start_point, direction, img_size):
        dir_x, dir_y = direction[0].item(), direction[1].item()
        ratios = [float('inf')] * 4

        if dir_x > 0:
            ratios[2] = (img_size - 1 - start_point[0]) / dir_x
        elif dir_x < 0:
            ratios[3] = -start_point[0] / dir_x

        if dir_y > 0:
            ratios[0] = (img_size - 1 - start_point[1]) / dir_y
        elif dir_y < 0:
            ratios[1] = -start_point[1] / dir_y

        min_ratio = min(ratios)
        end_point = start_point + min_ratio * direction

        if end_point[0] > 511:
            end_point[0] = 511
        elif end_point[0] < 0:
            end_point[0] = 0
        if end_point[1] > 511:
            end_point[1] = 511
        elif end_point[1] < 0:
            end_point[1] = 0

        return end_point

    def calculate_startpoint(self, contour):
        idx = contour.shape[0] // 2
        start_point = torch.tensor(contour[idx].squeeze(), dtype=torch.float32)
        if start_point[0] > 511:
            start_point[0] = 511
        elif start_point[0] < 0:
            start_point[0] = 0
        if start_point[1] > 511:
            start_point[1] = 511
        elif start_point[1] < 0:
            start_point[1] = 0
        return start_point

    def add_lines_along_tail(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        tail_contours = self.find_contours(mask[1, :, :])
        head_contours = self.find_contours(mask[2, :, :])


        linesNumber = random.randint(0, 3)
        number = 1
        max_value = 511

        for contour, t_contour in zip(head_contours, tail_contours):
            if contour.shape[0] > 10 and not None and number <= linesNumber:

                start_point = self.calculate_startpoint(contour)
                idx2 = t_contour.shape[0] // 8
                if idx2 + 1 >= t_contour.shape[0]:
                    maximum = idx2
                else:
                    maximum = idx2 + 1
                if idx2 - 1 <= 0:
                    minimum = 0
                else:
                    minimum = idx2 - 1
                direction = torch.tensor(t_contour[maximum] - t_contour[minimum], dtype=torch.float32).squeeze()
                direction = direction / torch.norm(direction)  
                end_point1 = self.calculate_endpoint(start_point, direction, 511)
                end_point2 = self.calculate_endpoint(start_point, -direction, 511)
                end_point3 = self.calculate_endpoint(start_point - 1, direction, 511)
                end_point4 = self.calculate_endpoint(start_point - 1, -direction, 511)
                end_point5 = self.calculate_endpoint(start_point - 2, direction, 511)
                end_point6 = self.calculate_endpoint(start_point - 2, -direction, 511)

                color_line = random.uniform(0.2, 0.4)
                img_np = np.array(img)
                mean_color = np.mean(img_np) * color_line
                color = torch.tensor([mean_color, mean_color, mean_color])
                mask = torch.zeros_like(img)
                if not math.isnan(end_point1[0]) and not math.isnan(end_point1[1]) and not math.isnan(
                        end_point2[0]) and not math.isnan(end_point2[1]):
                    if 0 <= start_point[0] <= max_value and 0 <= start_point[1] <= max_value:
                        mask = K.utils.draw_line(mask, start_point, end_point1, color)
                        mask = K.utils.draw_line(mask, start_point, end_point2, color)
                    if 0 <= start_point[0] - 1 <= max_value and 0 <= start_point[1] - 1 <= max_value:
                        mask = K.utils.draw_line(mask, start_point - 1, end_point3, color)
                        mask = K.utils.draw_line(mask, start_point - 1, end_point4, color)
                    if 0 <= start_point[0] - 2 <= max_value and 0 <= start_point[1] - 2 <= max_value:
                        mask = K.utils.draw_line(mask, start_point - 2, end_point5, color)
                        mask = K.utils.draw_line(mask, start_point - 2, end_point6, color)

                    number += 1

                max = 35
                min = 25
                kernel = random.randint(min, max)
                if kernel % 2 == 0:
                    kernel += 1

                mask_np = mask.numpy()
                mask_np = cv2.GaussianBlur(mask_np, (kernel, kernel), 0)

                mask_blurred = torch.from_numpy(mask_np)

                operation = random.choice(['add', 'subtract'])
                if operation == 'add':
                    img = img + mask_blurred
                else:
                    img = img - mask_blurred
        return img

    def change_brightness(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tail_contours = self.find_contours(mask[1, :, :])
        head_contours = self.find_contours(mask[2, :, :])


        linesNumber = random.randint(0, 1)

        number = 1

        for contour, t_contour in zip(head_contours, tail_contours):
            if contour.shape[0] > 10 and not None and number <= linesNumber:

                start_point = self.calculate_startpoint(contour)
                idx2 = t_contour.shape[0] // 8
                if idx2 + 1 >= t_contour.shape[0]:
                    maximum = idx2
                else:
                    maximum = idx2 + 1
                if idx2 - 1 <= 0:
                    minimum = 0
                else:
                    minimum = idx2 - 1
                direction = torch.tensor(t_contour[maximum] - t_contour[minimum], dtype=torch.float32).squeeze()
                direction = direction / torch.norm(direction) 
                end_point1 = self.calculate_endpoint(start_point, direction, 511)
                end_point2 = self.calculate_endpoint(start_point, -direction, 511)
                end_point3 = self.calculate_endpoint(start_point - 1, direction, 511)
                end_point4 = self.calculate_endpoint(start_point - 1, -direction, 511)
                end_point5 = self.calculate_endpoint(start_point - 2, direction, 511)
                end_point6 = self.calculate_endpoint(start_point - 2, -direction, 511)

                color_line = random.uniform(0.2, 0.4)
                img_np = np.array(img)
                mean_color = np.mean(img_np) * color_line
                color = torch.tensor([mean_color, mean_color, mean_color])
                mask = torch.zeros_like(img)
                max_value = 511
                if not math.isnan(end_point1[0]) and not math.isnan(end_point1[1]) and not math.isnan(
                        end_point2[0]) and not math.isnan(end_point2[1]):
                    if not math.isnan(end_point1[0]) and not math.isnan(end_point1[1]) and not math.isnan(
                            end_point2[0]) and not math.isnan(end_point2[1]):
                        if 0 <= start_point[0] <= max_value and 0 <= start_point[1] <= max_value:
                            mask = K.utils.draw_line(mask, start_point, end_point1, color)
                            mask = K.utils.draw_line(mask, start_point, end_point2, color)
                        if 0 <= start_point[0] - 1 <= max_value and 0 <= start_point[1] - 1 <= max_value:
                            mask = K.utils.draw_line(mask, start_point - 1, end_point3, color)
                            mask = K.utils.draw_line(mask, start_point - 1, end_point4, color)
                        if 0 <= start_point[0] - 2 <= max_value and 0 <= start_point[1] - 2 <= max_value:
                            mask = K.utils.draw_line(mask, start_point - 2, end_point5, color)
                            mask = K.utils.draw_line(mask, start_point - 2, end_point6, color)

                    number += 1

                maximum = 35
                minimum = 25
                kernel = random.randint(minimum, maximum)
                if kernel % 2 == 0:
                    kernel += 1


                mask_np = mask.numpy()
                mask_np = mask_np.transpose(1, 2, 0)
                mask_np = cv2.rotate(mask_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

                mask_np = cv2.dilate(mask_np, np.ones((100, 100), np.uint8), iterations=3)
                mask_np = cv2.GaussianBlur(mask_np, (kernel, kernel), 0)
                mask_np = mask_np.transpose(2, 0, 1)

                mask_blurred = torch.from_numpy(mask_np)

                operation = random.choice(['add', 'subtract'])
                if operation == 'add':
                    img = img + mask_blurred
                else:
                    img = img - mask_blurred

        return img

    def add_circles(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        circleNumber = random.randint(0, 5)
        for _ in range(circleNumber):
            overlap = True
            while overlap:
                center_coordinates = torch.randint(0, img.shape[1], (2,)).float()
                radius_x = torch.tensor(random.randint(5, 15)).float()
                radius_y = torch.tensor(random.randint(5, 15)).float()
                num_sides = 100
                theta = torch.linspace(0, 2 * np.pi, num_sides)
                x = center_coordinates[0] + radius_x * torch.cos(theta)
                y = center_coordinates[1] + radius_y * torch.sin(theta)

                x_clamped = torch.clamp(x.long(), 0, mask.shape[0] - 1)
                y_clamped = torch.clamp(y.long(), 0, mask.shape[1] - 1)
                sperm_mask = mask[3, :, :]
                overlap = torch.any(sperm_mask[y_clamped, x_clamped] > 0)


            polygon = torch.stack((x, y), dim=-1).unsqueeze(0)
            polygon2 = torch.stack((x - 1, y - 1), dim=-1).unsqueeze(0)
            mean_color = img.float().mean(dim=(1, 2))
            img = img.unsqueeze(0)

            color = mean_color + torch.tensor([0.45, 0.45, 0.45])
            color2 = torch.tensor([1, 1, 1])


            max = 17
            min = 9
            kernel = random.randint(min, max)
            if kernel % 2 == 0:
                kernel += 1

            mask1 = torch.zeros_like(img)
            mask2 = torch.zeros_like(img)
            mask2 = K.utils.draw_convex_polygon(mask1, polygon2, color2)
            mask1 = K.utils.draw_convex_polygon(mask1, polygon, color)

            mask1_np = mask1.numpy()
            mask1_np = mask1.squeeze().numpy()
            mask1_np = cv2.GaussianBlur(mask1_np, (kernel, kernel), 0)

            mask2_np = mask2.numpy()
            mask2_np = mask2.squeeze().numpy()
            mask2_np = cv2.GaussianBlur(mask2_np, (kernel, kernel), 0)

            mask1 = torch.from_numpy(mask1_np)
            mask2 = torch.from_numpy(mask2_np)

            img = img + mask1 - mask2

            img = img.squeeze(0)

        return img

    def add_blur_along_tail(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tail_contours = self.find_contours(mask[1, :, :])

        for contour in tail_contours:
            if contour.shape[0] > 10 and not None:

                segments = np.array_split(contour, 20)
                idx = random.randint(1, 4)
                segments_to_blur = random.sample(segments, k=idx)
                expand_by = 5
                for segment in segments_to_blur:
                    if segment.size > 0:
                        start_point = (np.min(segment[:, 0, 0] - expand_by), np.min(segment[:, 0, 1] - expand_by))
                        end_point = (np.max(segment[:, 0, 0] + expand_by), np.max(segment[:, 0, 1] + expand_by))


                        img = img.unsqueeze(0)
                        segment_img = img[:, :, start_point[1]:end_point[1], start_point[0]:end_point[0]].clone()

                        if min(segment_img.shape[2], segment_img.shape[3]) >= 5:
                            blurred_segment = K.filters.box_blur(segment_img, (10, 10))
                        else:
                            blurred_segment = segment_img
                        img[:, :, start_point[1]:end_point[1], start_point[0]:end_point[0]] = blurred_segment
                        img = img.squeeze(0)

        return img

    def rotate_contour(self, contour, angle, center):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        contour = contour.reshape(-1, 2)
        rotated_contour = cv2.transform(np.array([contour]), rotation_matrix)

        return rotated_contour[0]

    def copy_paste_sperm(self, img: torch.Tensor, mask: torch.Tensor, mask2: torch.Tensor, mask3: torch.Tensor) -> (
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        # Kopiowanie obrazu i masek
        new_img = img.clone()
        new_mask = mask.clone()
        new_mask2 = mask2.clone()
        new_mask3 = mask3.clone()

        i = 0
        j = 3

        max_count = random.randint(0, 3)

        cx_list = []
        cy_list = []
        angle_list = []
        translation_list = []

        for layer in range(mask.shape[0]):
            contours, _ = cv2.findContours(mask[j].byte().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(mask2[j].byte().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours3, _ = cv2.findContours(mask3[j].byte().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if layer == 0:
                continue

            processed_count = 0
            l = 0

            for contour, contour2, contour3 in zip(contours, contours2,
                                                   contours3): 

                if max_count is not None and processed_count >= max_count:
                    break

                if (cv2.contourArea(contour) > 10 and
                        cv2.contourArea(contour2) > 10 and
                        cv2.contourArea(contour3) > 10):  

                    moments = cv2.moments(contour)
                    moments2 = cv2.moments(contour2)
                    moments3 = cv2.moments(contour3)

                    if moments['m00'] == 0 or moments2['m00'] == 0 or moments3['m00'] == 0:
                        continue

                    if i < 1:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                        cx2 = int(moments2['m10'] / moments2['m00'])
                        cy2 = int(moments2['m01'] / moments2['m00'])
                        cx3 = int(moments3['m10'] / moments3['m00'])
                        cy3 = int(moments3['m01'] / moments3['m00'])

                        angle = random.randint(0, 360)
                        translation = random.randint(-20, -5)

                        cx_list.append(cx)
                        cy_list.append(cy)
                        angle_list.append(angle)
                        translation_list.append(translation)

                    if l > len(angle_list) - 1:
                        l = len(angle_list) - 1

                    rotated_contour = self.rotate_contour(contour, angle_list[l], (cx_list[l], cy_list[l]))
                    rotated_contour2 = self.rotate_contour(contour2, angle_list[l], (cx2, cy2))
                    rotated_contour3 = self.rotate_contour(contour3, angle_list[l], (cx3, cy3))

                    new_layer_mask = np.zeros_like(mask[j].numpy())
                    new_layer_mask2 = np.zeros_like(mask2[j].numpy())
                    new_layer_mask3 = np.zeros_like(mask3[j].numpy())

                    cv2.drawContours(new_layer_mask, [rotated_contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
                    cv2.drawContours(new_layer_mask2, [rotated_contour2.astype(np.int32)], -1, 255,
                                     thickness=cv2.FILLED)
                    cv2.drawContours(new_layer_mask3, [rotated_contour3.astype(np.int32)], -1, 255,
                                     thickness=cv2.FILLED)

                    rotation_matrix = cv2.getRotationMatrix2D((cx_list[l], cy_list[l]), angle_list[l], 1.0)
                    inv_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)

                    rotation_matrix2 = cv2.getRotationMatrix2D((cx2, cy2), angle_list[l], 1.0)
                    inv_rotation_matrix2 = cv2.invertAffineTransform(rotation_matrix2)

                    rotation_matrix3 = cv2.getRotationMatrix2D((cx3, cy3), angle_list[l], 1.0)
                    inv_rotation_matrix3 = cv2.invertAffineTransform(rotation_matrix3)

                    for y in range(new_layer_mask.shape[0]):
                        for x in range(new_layer_mask.shape[1]):
                            if new_layer_mask[y, x] == 255:
                                new_y = (y + translation_list[l]) % img.shape[1]
                                new_x = (x + translation_list[l]) % img.shape[2]
                                original_coords = np.dot(inv_rotation_matrix, np.array([x, y, 1]))
                                orig_x, orig_y = int(original_coords[0]), int(original_coords[1])
                                if 0 <= orig_y < img.shape[1] and 0 <= orig_x < img.shape[2]:
                                    new_img[:, new_y, new_x] = img[:, orig_y, orig_x]
                                    new_mask[j, new_y, new_x] = 1

                            if new_layer_mask2[y, x] == 255:
                                new_y2 = (y + translation_list[l]) % img.shape[1]
                                new_x2 = (x + translation_list[l]) % img.shape[2]
                                original_coords2 = np.dot(inv_rotation_matrix2, np.array([x, y, 1]))
                                orig_x2, orig_y2 = int(original_coords2[0]), int(original_coords2[1])
                                if 0 <= orig_y2 < img.shape[1] and 0 <= orig_x2 < img.shape[2]:
                                    new_mask2[j, new_y2, new_x2] = 1

                            if new_layer_mask3[y, x] == 255:
                                new_y3 = (y + translation_list[l]) % img.shape[1]
                                new_x3 = (x + translation_list[l]) % img.shape[2]
                                original_coords3 = np.dot(inv_rotation_matrix3, np.array([x, y, 1]))
                                orig_x3, orig_y3 = int(original_coords3[0]), int(original_coords3[1])
                                if 0 <= orig_y3 < img.shape[1] and 0 <= orig_x3 < img.shape[2]:
                                    new_mask3[j, new_y3, new_x3] = 1

                processed_count += 1
                l += 1

            j -= 1
            i += 1

        return new_img, new_mask, new_mask2, new_mask3







