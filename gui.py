import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QLineEdit,
                             QCheckBox, QComboBox, QStackedLayout, QFileDialog)
from PyQt5.QtCore import QProcess
import os


class TrainingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.training_params = {}  
        self.inference_params = {}  
        self.initUI()

    def initUI(self):
        self.process = QProcess()
        self.main_layout = QVBoxLayout()

        self.mode_layout = QHBoxLayout()
        self.training_button = QPushButton('Training')
        self.inference_button = QPushButton('Inference')
        self.training_button.clicked.connect(self.switch_to_training)
        self.inference_button.clicked.connect(self.switch_to_inference)
        self.mode_layout.addWidget(self.training_button)
        self.mode_layout.addWidget(self.inference_button)
        self.main_layout.addLayout(self.mode_layout)

        self.stacked_layout = QStackedLayout()
        self.main_layout.addLayout(self.stacked_layout)

        self.training_layout = QVBoxLayout()
        self.create_training_layout()

        self.inference_layout = QVBoxLayout()
        self.create_inference_layout()

        self.training_widget = QWidget()
        self.training_widget.setLayout(self.training_layout)
        self.inference_widget = QWidget()
        self.inference_widget.setLayout(self.inference_layout)

        self.stacked_layout.addWidget(self.training_widget)
        self.stacked_layout.addWidget(self.inference_widget)

        self.stacked_layout.setCurrentWidget(self.training_widget)

        self.setLayout(self.main_layout)
        self.setWindowTitle('Training and Inference Application')
        self.setGeometry(300, 300, 600, 400)

        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

    def create_training_layout(self):
        self.add_param_field(self.training_layout, 'epochs', 'Number of epochs:', '300', self.training_params)
        self.add_param_field(self.training_layout, 'batch_size', 'Batch size:', '6', self.training_params)
        self.add_param_field(self.training_layout, 'lr', 'Learning rate:', '1e-3', self.training_params)

        self.add_combo_field(self.training_layout, 'annotator', 'Annotator:', [str(i) for i in range(1, 6)],
                             default_value='2', params_dict=self.training_params)

        self.add_combo_field(self.training_layout, 'model', 'Model:',
                             ['smpUNet', 'smpUNet++', 'MAnet', 'DeepLabV3+', 'DualUNetPlusPlus', 'QuadUNetPlusPlus'],
                             default_value='smpUNet++', params_dict=self.training_params)
        self.add_combo_field(self.training_layout, 'loss', 'Loss function:',
                             ['CrossEntropyLoss', 'CrossEntropyLossWeight', 'BCEWithLogitsLoss', 'BCE'],
                             default_value='CrossEntropyLoss', params_dict=self.training_params)
        self.add_combo_field(self.training_layout, 'optimizer', 'Optimizer:', ['Adam', 'SGD', 'RMSprop'],
                             default_value='Adam', params_dict=self.training_params)
        self.add_combo_field(self.training_layout, 'scheduler', 'Scheduler:',
                             ['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'None'],
                             default_value='CosineAnnealingLR', params_dict=self.training_params)
        self.add_combo_field(self.training_layout, 'mode', 'Mode:',
                             ['intersection', 'intersection_and_union', 'cascade', 'cascade_con', 'two_task_training',
                              'two_task_training(4)'], default_value='intersection', params_dict=self.training_params)
        self.add_combo_field(self.training_layout, 'place', 'Place:', ['laptop', 'lab', 'komputer'],
                             default_value='lab', params_dict=self.training_params)
        self.add_combo_field(self.training_layout, 'aug_type', 'Augmentation Type:',
                             ['BasicAugmentation', 'ClassSpecificAugmentation'],
                             default_value='ClassSpecificAugmentation', params_dict=self.training_params)

        self.add_param_field(self.training_layout, 'k', 'K parameter:', '5', self.training_params)

        self.use_augmentation = QCheckBox('Use Augmentation')
        self.training_layout.addWidget(self.use_augmentation)

        self.start_button = QPushButton('Start Training')
        self.start_button.clicked.connect(self.start_training)
        self.training_layout.addWidget(self.start_button)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.training_layout.addWidget(self.log_output)

    def create_inference_layout(self):
        self.add_combo_field(self.inference_layout, 'model', 'Model:',
                             ['smpUNet', 'smpUNet++', 'DualUNetPlusPlus', 'QuadUNetPlusPlus'],
                             default_value='smpUNet++', params_dict=self.inference_params)
        self.add_combo_field(self.inference_layout, 'place', 'Place:', ['laptop', 'lab', 'komputer'],
                             default_value='lab', params_dict=self.inference_params)
        self.add_combo_field(self.inference_layout, 'mode', 'Mode:', [
            'intersection_and_union_inference', 'intersection_inference',
            'cascade', 'cascade_con', 'two_task_training','two_task_training(4)' ], default_value='intersection_inference', params_dict=self.inference_params)

        self.add_param_field(self.inference_layout, 'k', 'K parameter:', '20', self.inference_params)
        self.add_combo_field(self.inference_layout, 'annotator', 'Annotator:', [str(i) for i in range(1, 4)],
                             default_value='1', params_dict=self.inference_params)

        file_layout = QHBoxLayout()
        self.model_path_field = QLineEdit(self)
        self.browse_button = QPushButton('Browse')
        self.browse_button.clicked.connect(self.browse_model_file)
        file_layout.addWidget(QLabel('Model file:'))
        file_layout.addWidget(self.model_path_field)
        file_layout.addWidget(self.browse_button)
        self.inference_layout.addLayout(file_layout)

        self.inference_button_start = QPushButton('Start Inference')
        self.inference_button_start.clicked.connect(self.start_inference)
        self.inference_layout.addWidget(self.inference_button_start)

        self.log_output_inference = QTextEdit()
        self.log_output_inference.setReadOnly(True)
        self.inference_layout.addWidget(self.log_output_inference)

    def browse_model_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "All Files (*)")
        if file_name:
            model_name = os.path.basename(file_name)
            self.model_path_field.setText(model_name)

    def add_param_field(self, layout, name, label_text, default_value, params_dict):
        label = QLabel(label_text)
        line_edit = QLineEdit()
        line_edit.setText(default_value)
        layout.addWidget(label)
        layout.addWidget(line_edit)
        params_dict[name] = line_edit

    def add_combo_field(self, layout, name, label_text, options, default_value=None, params_dict=None):
        label = QLabel(label_text)
        combo = QComboBox()
        combo.addItems(options)
        if default_value:
            combo.setCurrentText(default_value)
        layout.addWidget(label)
        layout.addWidget(combo)
        params_dict[name] = combo

    def start_training(self):
        args = []
        for name, widget in self.training_params.items():
            if isinstance(widget, QLineEdit):
                args.append(f'--{name}')
                args.append(widget.text())
            elif isinstance(widget, QComboBox):
                args.append(f'--{name}')
                args.append(widget.currentText())

        if self.use_augmentation.isChecked():
            args.append('--augmentation')

        self.process.start('python3', ['train.py'] + args)

    def start_inference(self):
        args = []
        for name, widget in self.inference_params.items():
            if isinstance(widget, QLineEdit):
                args.append(f'--{name}')
                args.append(widget.text())
            elif isinstance(widget, QComboBox):
                args.append(f'--{name}')
                args.append(widget.currentText())

        args.append('--save_model_name')
        args.append(self.model_path_field.text())

        self.process.start('python', ['inference.py'] + args)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode('utf8')
        if self.stacked_layout.currentWidget() == self.training_widget:
            self.log_output.append(text)
        else:
            self.log_output_inference.append(text)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        text = bytes(data).decode('utf8')
        if self.stacked_layout.currentWidget() == self.training_widget:
            self.log_output.append(text)
        else:
            self.log_output_inference.append(text)

    def process_finished(self):
        if self.stacked_layout.currentWidget() == self.training_widget:
            self.log_output.append("Training finished.")
        else:
            self.log_output_inference.append("Inference finished.")

    def switch_to_training(self):
        self.stacked_layout.setCurrentWidget(self.training_widget)
        self.log_output.append("Switched to Training Mode.")

    def switch_to_inference(self):
        self.stacked_layout.setCurrentWidget(self.inference_widget)
        self.log_output_inference.clear()
        self.log_output_inference.append("Switched to Inference Mode.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TrainingApp()
    ex.show()
    sys.exit(app.exec_())
