import sys
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QLineEdit, QComboBox,
    QFileDialog, QTabWidget, QSlider
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from models.DualUNetPlusPlus import DualUNetPlusPlus
from models.QuadUNetPlusPlus import QuadUNetPlusPlus
import segmentation_models_pytorch as smp
from utils.metrics import calculate_iou, calculate_ap_for_segmentation


def rgb_to_class_map(rgb_mask):
    class_map = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)

    class_map[(rgb_mask[:, :, 0] == 0) & (rgb_mask[:, :, 1] == 255) & (rgb_mask[:, :, 2] == 0)] = 1

    class_map[(rgb_mask[:, :, 0] == 255) & (rgb_mask[:, :, 1] == 0) & (rgb_mask[:, :, 2] == 0)] = 2

    return class_map


class PredictApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout()

        self.model_dict = {
            'smpUNet++': smp.UnetPlusPlus(in_channels=3, classes=4, encoder_name="resnet18", encoder_weights=None),
            'smpUNet': smp.Unet(in_channels=3, classes=4),
            'DualUNetPlusPlus': DualUNetPlusPlus(3, 4),
            'QuadUNetPlusPlus': QuadUNetPlusPlus(3, 4)
        }

        self.tabs = QTabWidget()
        self.tab_predict = QWidget()
        self.tab_predict_with_gt = QWidget()
        self.tabs.addTab(self.tab_predict, "Predict")
        self.tabs.addTab(self.tab_predict_with_gt, "Predict with GT")

        self.tab_predict_layout = QVBoxLayout()
        self.add_predict_controls()
        self.tab_predict.setLayout(self.tab_predict_layout)

        self.tab_predict_with_gt_layout = QVBoxLayout()
        self.add_predict_with_gt_controls()
        self.tab_predict_with_gt.setLayout(self.tab_predict_with_gt_layout)

        self.main_layout.addWidget(self.tabs)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.main_layout.addWidget(self.log_output)

        self.setLayout(self.main_layout)
        self.setWindowTitle('Prediction Application')
        self.setGeometry(300, 300, 1200, 800)

    def add_predict_controls(self):
        self.model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_dict.keys())
        self.tab_predict_layout.addWidget(self.model_label)
        self.tab_predict_layout.addWidget(self.model_combo)

        self.model_file_button = QPushButton("Select Model File")
        self.model_file_button.clicked.connect(self.select_model_file)
        self.model_file_path = QLineEdit()
        self.tab_predict_layout.addWidget(self.model_file_button)
        self.tab_predict_layout.addWidget(self.model_file_path)

        self.image_button = QPushButton("Select Image")
        self.image_button.clicked.connect(self.select_image)
        self.image_path = QLineEdit()
        self.tab_predict_layout.addWidget(self.image_button)
        self.tab_predict_layout.addWidget(self.image_path)

        self.threshold_head_label = QLabel("Threshold Head:")
        self.threshold_head_slider = QSlider(Qt.Horizontal)
        self.threshold_head_slider.setMinimum(0)
        self.threshold_head_slider.setMaximum(255)
        self.threshold_head_slider.setValue(118)
        self.threshold_head_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_head_slider.setTickInterval(10)
        self.threshold_head_slider.valueChanged.connect(self.update_head_threshold_text)

        self.threshold_head_value = QLineEdit()
        self.threshold_head_value.setText(str(self.threshold_head_slider.value()))
        self.threshold_head_value.setFixedWidth(50)
        self.threshold_head_value.editingFinished.connect(self.update_head_threshold_slider)

        head_layout = QHBoxLayout()
        head_layout.addWidget(self.threshold_head_label)
        head_layout.addWidget(self.threshold_head_slider)
        head_layout.addWidget(self.threshold_head_value)
        self.tab_predict_layout.addLayout(head_layout)

        self.threshold_tail_label = QLabel("Threshold Tail:")
        self.threshold_tail_slider = QSlider(Qt.Horizontal)
        self.threshold_tail_slider.setMinimum(0)
        self.threshold_tail_slider.setMaximum(255)
        self.threshold_tail_slider.setValue(162)
        self.threshold_tail_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_tail_slider.setTickInterval(10)
        self.threshold_tail_slider.valueChanged.connect(self.update_tail_threshold_text)

        self.threshold_tail_value = QLineEdit()
        self.threshold_tail_value.setText(str(self.threshold_tail_slider.value()))
        self.threshold_tail_value.setFixedWidth(50)
        self.threshold_tail_value.editingFinished.connect(self.update_tail_threshold_slider)

        tail_layout = QHBoxLayout()
        tail_layout.addWidget(self.threshold_tail_label)
        tail_layout.addWidget(self.threshold_tail_slider)
        tail_layout.addWidget(self.threshold_tail_value)
        self.tab_predict_layout.addLayout(tail_layout)

        self.predict_button = QPushButton("Run Prediction")
        self.predict_button.clicked.connect(self.run_prediction)
        self.tab_predict_layout.addWidget(self.predict_button)

        self.result_tabs = QTabWidget()
        self.tab_argmax = QWidget()
        self.tab_softmask = QWidget()
        self.tab_custom = QWidget()
        self.result_tabs.addTab(self.tab_argmax, "Argmax Mask")
        self.result_tabs.addTab(self.tab_softmask, "Softmask")
        self.result_tabs.addTab(self.tab_custom, "Custom Threshold Mask")

        self.tab_predict_layout.addWidget(self.result_tabs)

        self.input_image_display_argmax, self.prediction_image_display_argmax, _ = self.add_image_layout(
            self.tab_argmax)
        self.input_image_display_softmask, self.prediction_image_display_softmask, _ = self.add_image_layout(
            self.tab_softmask)
        self.input_image_display_custom, self.prediction_image_display_custom, _ = self.add_image_layout(
            self.tab_custom)

    def update_custom_mask(self):
        if not hasattr(self, 'softs1') or not hasattr(self, 'image_original'):
            return
        threshold_head = self.threshold_head_slider.value() / 255
        threshold_tail = self.threshold_tail_slider.value() / 255
        threshold_oneclass = 80.0

        final_mask, _ = self.apply_thresholds(self.softs1, self.softs1, threshold_head, threshold_tail,
                                              threshold_oneclass)

        colored_final_mask = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        colored_final_mask[final_mask == 1] = [0, 255, 0]
        colored_final_mask[final_mask == 2] = [255, 0, 0]

        self.display_image(self.image_original, self.input_image_display_custom)
        self.display_image(colored_final_mask, self.prediction_image_display_custom)

    def update_head_threshold_text(self):
        self.threshold_head_value.setText(str(self.threshold_head_slider.value()))
        self.update_custom_mask()

    def update_head_threshold_slider(self):
        value = int(self.threshold_head_value.text())
        self.threshold_head_slider.setValue(value)

    def update_tail_threshold_text(self):
        self.threshold_tail_value.setText(str(self.threshold_tail_slider.value()))
        self.update_custom_mask()

    def update_tail_threshold_slider(self):
        value = int(self.threshold_tail_value.text())
        self.threshold_tail_slider.setValue(value)

    def add_predict_with_gt_controls(self):
        self.model_label_gt = QLabel("Model:")
        self.model_combo_gt = QComboBox()
        self.model_combo_gt.addItems(self.model_dict.keys())
        self.tab_predict_with_gt_layout.addWidget(self.model_label_gt)
        self.tab_predict_with_gt_layout.addWidget(self.model_combo_gt)

        self.model_file_button = QPushButton("Select Model File")
        self.model_file_button.clicked.connect(self.select_model_file)
        self.model_file_path = QLineEdit()
        self.tab_predict_with_gt_layout.addWidget(self.model_file_button)
        self.tab_predict_with_gt_layout.addWidget(self.model_file_path)

        self.image_button_gt = QPushButton("Select Image")
        self.image_button_gt.clicked.connect(self.select_image_gt)
        self.image_path_gt = QLineEdit()
        self.tab_predict_with_gt_layout.addWidget(self.image_button_gt)
        self.tab_predict_with_gt_layout.addWidget(self.image_path_gt)

        self.gt_button = QPushButton("Select Ground Truth")
        self.gt_button.clicked.connect(self.select_gt)
        self.gt_path = QLineEdit()
        self.tab_predict_with_gt_layout.addWidget(self.gt_button)
        self.tab_predict_with_gt_layout.addWidget(self.gt_path)

        self.threshold_head_label_gt = QLabel("Threshold Head:")
        self.threshold_head_slider_gt = QSlider(Qt.Horizontal)
        self.threshold_head_slider_gt.setMinimum(0)
        self.threshold_head_slider_gt.setMaximum(255)
        self.threshold_head_slider_gt.setValue(118)
        self.threshold_head_slider_gt.setTickPosition(QSlider.TicksBelow)
        self.threshold_head_slider_gt.setTickInterval(10)
        self.threshold_head_slider_gt.valueChanged.connect(self.update_head_threshold_text_gt)

        self.threshold_head_value_gt = QLineEdit()
        self.threshold_head_value_gt.setText(str(self.threshold_head_slider_gt.value()))
        self.threshold_head_value_gt.setFixedWidth(50)
        self.threshold_head_value_gt.editingFinished.connect(self.update_head_threshold_slider_gt)

        head_layout_gt = QHBoxLayout()
        head_layout_gt.addWidget(self.threshold_head_label_gt)
        head_layout_gt.addWidget(self.threshold_head_slider_gt)
        head_layout_gt.addWidget(self.threshold_head_value_gt)
        self.tab_predict_with_gt_layout.addLayout(head_layout_gt)

        self.threshold_tail_label_gt = QLabel("Threshold Tail:")
        self.threshold_tail_slider_gt = QSlider(Qt.Horizontal)
        self.threshold_tail_slider_gt.setMinimum(0)
        self.threshold_tail_slider_gt.setMaximum(255)
        self.threshold_tail_slider_gt.setValue(162)
        self.threshold_tail_slider_gt.setTickPosition(QSlider.TicksBelow)
        self.threshold_tail_slider_gt.setTickInterval(10)
        self.threshold_tail_slider_gt.valueChanged.connect(self.update_tail_threshold_text_gt)

        self.threshold_tail_value_gt = QLineEdit()
        self.threshold_tail_value_gt.setText(str(self.threshold_tail_slider_gt.value()))
        self.threshold_tail_value_gt.setFixedWidth(50)
        self.threshold_tail_value_gt.editingFinished.connect(self.update_tail_threshold_slider_gt)

        tail_layout_gt = QHBoxLayout()
        tail_layout_gt.addWidget(self.threshold_tail_label_gt)
        tail_layout_gt.addWidget(self.threshold_tail_slider_gt)
        tail_layout_gt.addWidget(self.threshold_tail_value_gt)
        self.tab_predict_with_gt_layout.addLayout(tail_layout_gt)

        self.predict_button_gt = QPushButton("Run Prediction with GT")
        self.predict_button_gt.clicked.connect(self.run_prediction_with_gt)
        self.tab_predict_with_gt_layout.addWidget(self.predict_button_gt)

        self.result_tabs_gt = QTabWidget()
        self.tab_argmax_gt = QWidget()
        self.tab_softmask_gt = QWidget()
        self.tab_custom_gt = QWidget()
        self.result_tabs_gt.addTab(self.tab_argmax_gt, "Argmax Mask")
        self.result_tabs_gt.addTab(self.tab_softmask_gt, "Softmask")
        self.result_tabs_gt.addTab(self.tab_custom_gt, "Custom Threshold Mask")

        self.tab_predict_with_gt_layout.addWidget(self.result_tabs_gt)

        self.input_image_display_argmax_gt, self.prediction_image_display_argmax_gt, self.gt_image_display_argmax_gt = self.add_image_layout(
            self.tab_argmax_gt, include_gt=True)
        self.input_image_display_softmask_gt, self.prediction_image_display_softmask_gt, self.gt_image_display_softmask_gt = self.add_image_layout(
            self.tab_softmask_gt, include_gt=True)
        self.input_image_display_custom_gt, self.prediction_image_display_custom_gt, self.gt_image_display_custom_gt = self.add_image_layout(
            self.tab_custom_gt, include_gt=True)


    def update_head_threshold_text_gt(self):
        self.threshold_head_value_gt.setText(str(self.threshold_head_slider_gt.value()))
        self.update_custom_mask_gt()

    def update_head_threshold_slider_gt(self):
        value = int(self.threshold_head_value_gt.text())
        self.threshold_head_slider_gt.setValue(value)
        self.update_custom_mask_gt()

    def update_tail_threshold_text_gt(self):
        self.threshold_tail_value_gt.setText(str(self.threshold_tail_slider_gt.value()))
        self.update_custom_mask_gt()

    def update_tail_threshold_slider_gt(self):
        value = int(self.threshold_tail_value_gt.text())
        self.threshold_tail_slider_gt.setValue(value)
        self.update_custom_mask_gt()

    def update_head_threshold_text(self):
        self.threshold_head_value.setText(str(self.threshold_head_slider.value()))
        self.update_custom_mask()

    def update_head_threshold_slider(self):
        value = int(self.threshold_head_value.text())
        self.threshold_head_slider.setValue(value)
        self.update_custom_mask()

    def update_tail_threshold_text(self):
        self.threshold_tail_value.setText(str(self.threshold_tail_slider.value()))
        self.update_custom_mask()

    def update_tail_threshold_slider(self):
        value = int(self.threshold_tail_value.text())
        self.threshold_tail_slider.setValue(value)
        self.update_custom_mask()

    def add_image_layout(self, tab_widget, include_gt=False):
        layout = QHBoxLayout()
        tab_widget.setLayout(layout)

        input_layout = QVBoxLayout()
        input_image_label = QLabel("Input Image")
        input_image_display = QLabel()
        input_layout.addWidget(input_image_label)
        input_layout.addWidget(input_image_display)
        layout.addLayout(input_layout)

        prediction_layout = QVBoxLayout()
        prediction_image_label = QLabel("Prediction")
        prediction_image_display = QLabel()
        prediction_layout.addWidget(prediction_image_label)
        prediction_layout.addWidget(prediction_image_display)
        layout.addLayout(prediction_layout)

        gt_image_display = None
        if include_gt:
            gt_layout = QVBoxLayout()
            gt_image_label = QLabel("Ground Truth")
            gt_image_display = QLabel()
            gt_layout.addWidget(gt_image_label)
            gt_layout.addWidget(gt_image_display)
            layout.addLayout(gt_layout)

        return input_image_display, prediction_image_display, gt_image_display

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path.setText(file_name)

    def select_image_gt(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.image_path_gt.setText(file_name)

    def select_gt(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Ground Truth", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.gt_path.setText(file_name)

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        img_original = img.copy()
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)
        img = (img - min_val) / (max_val - min_val)
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        return img, img_original

    def apply_thresholds(self, softmask_multiclass_np, softmask_oneclass_np, threshold_head, threshold_tail,
                         threshold_oneclass):
        pred_head = (softmask_multiclass_np[:, :, 2] > threshold_head).astype(float)
        pred_tail = (softmask_multiclass_np[:, :, 1] > threshold_tail).astype(float)

        final_mask = np.zeros_like(pred_head)
        conflict = (pred_head == 1) & (pred_tail == 1)
        tail_conflict_values = softmask_multiclass_np[:, :, 1][conflict]
        head_conflict_values = softmask_multiclass_np[:, :, 2][conflict]
        final_mask[conflict] = 2 - (tail_conflict_values > head_conflict_values).astype(int)
        final_mask[(pred_tail == 1) & ~conflict] = 1
        final_mask[(pred_head == 1) & ~conflict] = 2

        pred_oneclass = (softmask_oneclass_np[:, :, 1] > threshold_oneclass).astype(float)
        return final_mask, pred_oneclass

    def update_custom_mask(self):
        threshold_head = self.threshold_head_slider.value() / 255
        threshold_tail = self.threshold_tail_slider.value() / 255
        threshold_oneclass = 80.0
        final_mask, _ = self.apply_thresholds(self.softs1, self.softs1, threshold_head, threshold_tail,
                                              threshold_oneclass)
        colored_final_mask = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        colored_final_mask[final_mask == 1] = [0, 255, 0]
        colored_final_mask[final_mask == 2] = [255, 0, 0]
        self.display_image(self.image_original, self.input_image_display_custom)
        self.display_image(colored_final_mask, self.prediction_image_display_custom)

    def update_custom_mask_gt(self):
        if not hasattr(self, 'softs1') or not hasattr(self, 'image_original'):
            return
        threshold_head = self.threshold_head_slider_gt.value() / 255
        threshold_tail = self.threshold_tail_slider_gt.value() / 255
        threshold_oneclass = 80.0

        final_mask, _ = self.apply_thresholds(self.softs1, self.softs1, threshold_head, threshold_tail,
                                              threshold_oneclass)

        colored_final_mask = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        colored_final_mask[final_mask == 1] = [0, 255, 0]
        colored_final_mask[final_mask == 2] = [255, 0, 0]

        self.display_image(self.image_original, self.input_image_display_custom_gt)
        self.display_image(colored_final_mask, self.prediction_image_display_custom_gt)

    def run_prediction(self):
        model_name = self.model_combo.currentText()
        image_path = self.image_path.text()
        model_file_path = self.model_file_path.text()

        if not image_path:
            self.log_output.append("Please select an image file.")
            return

        image, image_original = self.load_image(image_path)

        model_class = self.model_dict[model_name]
        model = model_class
        model.load_state_dict(torch.load(model_file_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        image_tensor = torch.from_numpy(image).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)

        output1 = outputs[:, :3, :, :]
        output2 = outputs[:, [0, -1], :, :]

        preds1 = torch.argmax(output1, dim=1).squeeze(0).cpu().detach().numpy()

        colored_mask = np.zeros((preds1.shape[0], preds1.shape[1], 3), dtype=np.uint8)
        colored_mask[preds1 == 1] = [0, 255, 0]
        colored_mask[preds1 == 2] = [255, 0, 0]

        self.display_image(image_original, self.input_image_display_argmax)
        self.display_image(colored_mask, self.prediction_image_display_argmax)

        softs1 = torch.softmax(output1, dim=1).squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        self.softs1 = softs1
        self.image_original = image_original

        softmask_combined = np.zeros((softs1.shape[0], softs1.shape[1], 3), dtype=np.uint8)
        softmask_combined[:, :, 0] = (softs1[:, :, 2] * 255).astype(np.uint8)
        softmask_combined[:, :, 1] = (softs1[:, :, 1] * 255).astype(np.uint8)

        self.display_image(image_original, self.input_image_display_softmask)
        self.display_image(softmask_combined, self.prediction_image_display_softmask)

        threshold_head = self.threshold_head_slider.value() / 255
        threshold_tail = self.threshold_tail_slider.value() / 255

        threshold_oneclass = 80.0

        final_mask, _ = self.apply_thresholds(softs1, softs1, threshold_head, threshold_tail, threshold_oneclass)

        colored_final_mask = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        colored_final_mask[final_mask == 1] = [0, 255, 0]
        colored_final_mask[final_mask == 2] = [255, 0, 0]

        self.display_image(image_original, self.input_image_display_custom)
        self.display_image(colored_final_mask, self.prediction_image_display_custom)

        self.log_output.append("Prediction completed.")

    def run_prediction_with_gt(self):
        model_name = self.model_combo_gt.currentText()
        image_path = self.image_path_gt.text()
        gt_path = self.gt_path.text()
        model_file_path = self.model_file_path.text()

        if not image_path or not gt_path:
            self.log_output.append("Please select an image and ground truth file.")
            return

        image, image_original = self.load_image(image_path)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.resize(gt_mask, (image.shape[3], image.shape[2]))
        gt_class_map = rgb_to_class_map(gt_mask)

        model_class = self.model_dict[model_name]
        model = model_class
        model.load_state_dict(torch.load(model_file_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        image_tensor = torch.from_numpy(image).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)

        output1 = outputs[:, :3, :, :]
        output2 = outputs[:, [0, -1], :, :]

        preds1 = torch.argmax(output1, dim=1).squeeze(0).cpu().detach().numpy()

        colored_mask = np.zeros((preds1.shape[0], preds1.shape[1], 3), dtype=np.uint8)
        colored_mask[preds1 == 1] = [0, 255, 0]
        colored_mask[preds1 == 2] = [255, 0, 0]

        self.display_image(image_original, self.input_image_display_argmax_gt)
        self.display_image(colored_mask, self.prediction_image_display_argmax_gt)
        self.display_image(gt_mask, self.gt_image_display_argmax_gt)

        softs1 = torch.softmax(output1, dim=1).squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        self.softs1 = softs1
        self.image_original = image_original

        softmask_combined = np.zeros((softs1.shape[0], softs1.shape[1], 3), dtype=np.uint8)
        softmask_combined[:, :, 0] = (softs1[:, :, 2] * 255).astype(np.uint8)
        softmask_combined[:, :, 1] = (softs1[:, :, 1] * 255).astype(np.uint8)

        self.display_image(image_original, self.input_image_display_softmask_gt)
        self.display_image(softmask_combined, self.prediction_image_display_softmask_gt)
        self.display_image(gt_mask, self.gt_image_display_softmask_gt)

        mean_iou, iou_per_class = calculate_iou(gt_class_map, preds1, 3)

        thresholds = torch.linspace(0, 1, steps=255)
        iou_scores_head = []
        iou_scores_tail = []

        for threshold in thresholds:
            pred_masks_multiclass = (softs1 > threshold.numpy()).astype(float)
            pred_masks_head = pred_masks_multiclass[:, :, 2]
            pred_masks_tail = pred_masks_multiclass[:, :, 1]

            iou_head, _ = calculate_iou(pred_masks_head, (gt_class_map == 2).astype(np.uint8), num_classes=2)
            iou_tail, _ = calculate_iou(pred_masks_tail, (gt_class_map == 1).astype(np.uint8), num_classes=2)

            iou_scores_head.append(iou_head)
            iou_scores_tail.append(iou_tail)

        optimal_threshold_head = thresholds[torch.argmax(torch.tensor(iou_scores_head))]
        optimal_threshold_tail = thresholds[torch.argmax(torch.tensor(iou_scores_tail))]

        optimal_iou_head = max(iou_scores_head)
        optimal_iou_tail = max(iou_scores_tail)

        ap_score_head = calculate_ap_for_segmentation(softs1[:, :, 2], (gt_class_map == 2).astype(np.uint8))
        ap_score_tail = calculate_ap_for_segmentation(softs1[:, :, 1], (gt_class_map == 1).astype(np.uint8))

        pred_head = (softs1[:, :, 2] > optimal_threshold_head.numpy()).astype(float)
        pred_tail = (softs1[:, :, 1] > optimal_threshold_tail.numpy()).astype(float)

        final_mask = np.zeros_like(pred_head)

        conflict = (pred_head == 1) & (pred_tail == 1)
        tail_conflict_values = softs1[:, :, 1][conflict]
        head_conflict_values = softs1[:, :, 2][conflict]

        final_mask[conflict] = 2 - (tail_conflict_values > head_conflict_values).astype(int)
        final_mask[(pred_tail == 1) & ~conflict] = 1
        final_mask[(pred_head == 1) & ~conflict] = 2

        final_iou, iou_per_class2 = calculate_iou(final_mask, gt_class_map, num_classes=3)

        self.log_output.append(f"Mean IoU for argmax prediction: {mean_iou}")
        self.log_output.append(f"Mean IoU for argmax (tail): {iou_per_class[1]:.4f}")
        self.log_output.append(f"Mean IoU for argmax (head): {iou_per_class[2]:.4f}")

        self.log_output.append(f"Optimal IoU (head): {optimal_iou_head:.4f} at threshold {int(optimal_threshold_head * 255)}")
        self.log_output.append(f"Optimal IoU (tail): {optimal_iou_tail:.4f} at threshold {int(optimal_threshold_tail * 255)}")

        self.log_output.append(f"Final Optimal IoU (combined head and tail): {final_iou:.4f}")

        self.log_output.append(f"Average Precision (AP) for head: {ap_score_head:.4f}")
        self.log_output.append(f"Average Precision (AP) for tail: {ap_score_tail:.4f}")

        threshold_head = self.threshold_head_slider_gt.value() / 255
        threshold_tail = self.threshold_tail_slider_gt.value() / 255
        threshold_oneclass = 80.0

        final_mask, _ = self.apply_thresholds(softs1, softs1, threshold_head, threshold_tail, threshold_oneclass)

        colored_final_mask = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        colored_final_mask[final_mask == 1] = [0, 255, 0]
        colored_final_mask[final_mask == 2] = [255, 0, 0]

        self.display_image(image_original, self.input_image_display_custom_gt)
        self.display_image(colored_final_mask, self.prediction_image_display_custom_gt)
        self.display_image(gt_mask, self.gt_image_display_custom_gt)

        self.log_output.append("Prediction with GT completed.")

    def display_image(self, image, label_widget):
        if len(image.shape) == 2:
            image = image.astype(np.uint8) * 255
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QPixmap(QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888))
        label_widget.setPixmap(q_image)

    def select_model_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "All Files (*)")
        if file_name:
            self.model_file_path.setText(file_name)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PredictApp()
    ex.show()
    sys.exit(app.exec_())
