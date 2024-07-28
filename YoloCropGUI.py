import sys
import os
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QCheckBox, QFileDialog, QMessageBox, QProgressBar, QComboBox, QCompleter, QToolTip, QScrollArea, QDialog, QFrame
from PyQt5.QtCore import Qt, QSettings, QStringListModel
import numpy as np
import colorsys
import shutil
import gc
from difflib import SequenceMatcher
from PyQt5.QtGui import QCloseEvent



def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def sort_models(model_list):
    return sorted(model_list, key=lambda x: (-max((similarity(x, y) for y in model_list if y != x), default=0), x))

def load_tags_from_model(model):
    if hasattr(model, 'names'):
        if isinstance(model.names, dict):
            return [name for _, name in sorted(model.names.items())]
        elif isinstance(model.names, (list, tuple)):
            return model.names
    elif hasattr(model, 'module') and hasattr(model.module, 'names'):
        return load_tags_from_model(model.module)
    else:
        print("Warning: Unable to extract class names from the model.")
        return []

class TagWidget(QWidget):
    def __init__(self, tag, parent=None):
        super().__init__(parent)
        self.tag = tag
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        label = QLabel(self.tag)
        layout.addWidget(label)

        remove_button = QPushButton("x")
        remove_button.setFixedWidth(20)
        remove_button.clicked.connect(self.remove_tag)
        layout.addWidget(remove_button)

    def remove_tag(self):
        self.parent().remove_tag(self.tag)

class CustomQCompleter(QCompleter):
    def __init__(self, tags_list, parent=None):
        super().__init__(tags_list, parent)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterMode(Qt.MatchContains)
        self.setCompletionMode(QCompleter.PopupCompletion)

    def splitPath(self, path):
        tags = path.split(' ')
        if len(tags) > 1 and tags[-1].strip() == '':
            tags = tags[:-1]
        return [tags[-1]]

    def pathFromIndex(self, index):
        return index.data()


class CropGUI(QWidget):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.settings = QSettings("YourCompany", "YourApp")
        self.tags_completer = None  # Initialize tags_completer
        self.skip_completer = None  # Initialize skip_completer
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Crop Around")
        main_layout = QHBoxLayout()  # Initialize main_layout
        self.stop_processing_flag = False

        # Main content layout
        content_layout = QVBoxLayout()  # Initialize content_layout

        # Model path
        model_layout = QHBoxLayout()
        model_label = QLabel("Model Path:")
        self.model_path_input = QComboBox()
        self.model_path_input.addItem("yolov8x-oiv7.pt")  # Add a default model
        self.load_models_from_directory()
        self.model_path_input.currentIndexChanged.connect(self.load_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_path_input)
        content_layout.addLayout(model_layout)  # Add to content_layout

        # Load the default model
        self.model = YOLO(self.model_path_input.currentText())
        self.tags_list = load_tags_from_model(self.model)

        # Device
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda:0", "cpu"])
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        content_layout.addLayout(device_layout)  # Add to content_layout

        # Source folder
        source_layout = QHBoxLayout()
        source_label = QLabel("Source Folder:")
        self.source_folder_input = QLineEdit(self.settings.value("source_folder", ""))
        self.source_folder_input.textChanged.connect(lambda: self.settings.setValue("source_folder", self.source_folder_input.text()))
        source_button = QPushButton("Browse")
        source_button.clicked.connect(self.browse_source_folder)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_folder_input)
        source_layout.addWidget(source_button)
        content_layout.addLayout(source_layout)  # Add to content_layout

        # Export folder
        export_layout = QHBoxLayout()
        export_label = QLabel("Export Folder:")
        self.export_folder_input = QLineEdit(self.settings.value("export_folder", ""))
        self.export_folder_input.textChanged.connect(lambda: self.settings.setValue("export_folder", self.export_folder_input.text()))
        export_button = QPushButton("Browse")
        export_button.clicked.connect(self.browse_export_folder)
        export_layout.addWidget(export_label)
        export_layout.addWidget(self.export_folder_input)
        export_layout.addWidget(export_button)
        content_layout.addLayout(export_layout)  # Add to content_layout

        # Single image path
        single_image_layout = QHBoxLayout()
        single_image_label = QLabel("Single Image Path:")
        self.single_image_input = QLineEdit(self.settings.value("single_image", ""))
        self.single_image_input.textChanged.connect(lambda: self.settings.setValue("single_image", self.single_image_input.text()))
        single_image_button = QPushButton("Browse")
        single_image_button.clicked.connect(self.browse_single_image)
        single_image_layout.addWidget(single_image_label)
        single_image_layout.addWidget(self.single_image_input)
        single_image_layout.addWidget(single_image_button)
        content_layout.addLayout(single_image_layout)  # Add to content_layout

        # Tags to crop
        tags_layout = QHBoxLayout()
        tags_label = QLabel("Tags to Crop:")
        self.tags_input = QLineEdit()
        self.tags_completer = CustomQCompleter(self.tags_list, self)
        self.tags_input.setCompleter(self.tags_completer)
        self.tags_input.returnPressed.connect(self.add_tag_to_crop)
        tags_layout.addWidget(tags_label)
        tags_layout.addWidget(self.tags_input)
        content_layout.addLayout(tags_layout)  # Add to content_layout

        self.tags_to_crop_layout = QVBoxLayout()
        content_layout.addLayout(self.tags_to_crop_layout)  # Add to content_layout

        # Skip if included
        skip_layout = QHBoxLayout()
        skip_label = QLabel("Skip if Included:")
        self.skip_input = QLineEdit()
        self.skip_completer = CustomQCompleter(self.tags_list, self)
        self.skip_input.setCompleter(self.skip_completer)
        self.skip_input.returnPressed.connect(self.add_tag_to_skip)
        skip_layout.addWidget(skip_label)
        skip_layout.addWidget(self.skip_input)
        content_layout.addLayout(skip_layout)  # Add to content_layout

        self.skip_if_included_layout = QVBoxLayout()
        content_layout.addLayout(self.skip_if_included_layout)  # Add to content_layout

        # Mark box
        mark_box_layout = QHBoxLayout()
        self.mark_box_checkbox = QCheckBox("Mark Box")
        mark_box_layout.addWidget(self.mark_box_checkbox)
        content_layout.addLayout(mark_box_layout)  # Add to content_layout

        # Ignore tags
        self.ignore_tags_checkbox = QCheckBox("Ignore Tags")
        mark_box_layout.addWidget(self.ignore_tags_checkbox)        

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        content_layout.addWidget(self.progress_bar)  # Add to content_layout

        # Start button
        self.process_button = QPushButton("Start")
        self.process_button.clicked.connect(self.process_images)
        content_layout.addWidget(self.process_button)  # Add to content_layout

        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)
        content_layout.addWidget(self.stop_button)  # Add to content_layout

        # Clear export folder button
        self.clear_export_button = QPushButton("Clear Export Folder")
        self.clear_export_button.clicked.connect(self.clear_export_folder)
        content_layout.addWidget(self.clear_export_button)  # Add to content_layout

        refresh_button = QPushButton("Refresh Models")
        refresh_button.clicked.connect(self.refresh_models)
        model_layout.addWidget(refresh_button)

        self.model_path_input.currentIndexChanged.connect(self.load_model)

        main_layout.addLayout(content_layout)

        # Side panel for model names
        self.model_names_panel = QFrame()
        self.model_names_panel.setFrameShape(QFrame.StyledPanel)
        self.model_names_panel.setFixedWidth(200)
        self.model_names_layout = QVBoxLayout(self.model_names_panel)
        self.model_names_scroll_area = QScrollArea()
        self.model_names_scroll_area.setWidgetResizable(True)
        self.model_names_content = QWidget()
        self.model_names_scroll_layout = QVBoxLayout(self.model_names_content)
        self.model_names_scroll_area.setWidget(self.model_names_content)
        self.model_names_layout.addWidget(self.model_names_scroll_area)
        self.model_names_panel.setVisible(True)  # Always visible

        main_layout.addWidget(self.model_names_panel)

        self.setLayout(main_layout)
        self.update_model_names_panel()

    def update_model_names_panel(self):
        if not hasattr(self, 'model') or not self.model:
            return

        model_names = load_tags_from_model(self.model)
        if not model_names:
            return

        # Clear the existing content
        for i in reversed(range(self.model_names_scroll_layout.count())):
            widget = self.model_names_scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Add new model names
        for i, name in enumerate(sorted(model_names), 1):
            label = QLabel(f"{i}. {name}")
            self.model_names_scroll_layout.addWidget(label)

    def load_model(self, index):
        model_path = self.model_path_input.currentText()
        if not model_path or model_path == "No models found":
            print("No valid model selected, returning from load_model.")
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Ensure the model path is not empty or invalid
            if not model_path or not os.path.exists(model_path):
                raise ValueError(f"Invalid model path: {model_path}")

            self.model = YOLO(model_path)
            self.tags_list = load_tags_from_model(self.model)
            self.update_tag_completer()

            # Save the current model as the last selected model
            self.settings.setValue("last_model", model_path)

            # Update model names panel
            self.update_model_names_panel()

        except Exception as e:
            print(f"Exception occurred: {e}")  # Debug print
            QMessageBox.warning(self, "Warning", f"Failed to load model: {str(e)}")


    def update_tag_completer(self):
        if self.tags_completer is None or self.skip_completer is None:
            print("Completers are not initialized yet.")
            return

        string_list_model = QStringListModel(self.tags_list)
        self.tags_completer.setModel(string_list_model)
        self.skip_completer.setModel(string_list_model)
        self.clear_tag_layouts()

    def load_models_from_directory(self):
        current_dir = os.getcwd()
        model_files = list(set([f for f in os.listdir(current_dir) if f.endswith('.pt') or f.endswith('.engine')]))
        
        if not model_files:
            QMessageBox.warning(self, "Warning", "No model files found in the directory.")
            self.model_path_input.addItem("No models found")
            return
        
        sorted_model_files = sort_models(model_files)
        
        self.model_path_input.clear()
        for model_file in sorted_model_files:
            print(f"Adding model file to combo box: {model_file}")
            self.model_path_input.addItem(model_file)
        
        last_model = self.settings.value("last_model", "")
        if last_model in sorted_model_files:
            index = self.model_path_input.findText(last_model)
            if index >= 0:
                self.model_path_input.setCurrentIndex(index)
        elif sorted_model_files:
            self.model_path_input.setCurrentIndex(0)
        else:
            self.model_path_input.addItem("No models found")

    def refresh_models(self):
        self.load_models_from_directory()
        if self.model_path_input.currentText() != "No models found":
            self.load_model(self.model_path_input.currentIndex())
        QMessageBox.information(self, "Refresh Complete", "Model list has been updated!")




    def display_model_names(self):
        if not hasattr(self, 'model') or not self.model:
            QMessageBox.warning(self, "Warning", "No model loaded.")
            return

        model_names = load_tags_from_model(self.model)
        if not model_names:
            QMessageBox.warning(self, "Warning", "No model names found.")
            return

        self.model_names_dialog = QDialog(self)
        self.model_names_dialog.setWindowTitle("Model Names")

        dialog_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for i, name in enumerate(sorted(model_names), 1):
            label = QLabel(f"{i}. {name}")
            scroll_layout.addWidget(label)

        scroll_area.setWidget(scroll_content)
        dialog_layout.addWidget(scroll_area)
        self.model_names_dialog.setLayout(dialog_layout)

        # Adjust the size of the dialog dynamically based on the main window size
        main_window_size = self.size()
        dialog_width = int(main_window_size.width() * 0.5)
        dialog_height = int(main_window_size.height() * 0.5)
        self.model_names_dialog.resize(dialog_width, dialog_height)

        self.model_names_dialog.show()


    def clear_export_folder(self):
        export_folder = self.export_folder_input.text()
        if os.path.exists(export_folder):
            for filename in os.listdir(export_folder):
                file_path = os.path.join(export_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            QMessageBox.information(self, "Clear Complete", "Export folder has been cleared!")
        else:
            QMessageBox.warning(self, "Folder Not Found", "Export folder does not exist!")

    @staticmethod
    def get_dynamic_parameters(image):
        height, width = image.shape[:2]
        diagonal = np.sqrt(height**2 + width**2)
        font_scale = diagonal / 1000
        line_thickness = max(1, int(diagonal / 500))
        text_height = int(height * 0.03)  # 3% of image height
        return font_scale, line_thickness, text_height


    def clear_tag_layouts(self):
        for i in reversed(range(self.tags_to_crop_layout.count())): 
            self.tags_to_crop_layout.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.skip_if_included_layout.count())): 
            self.skip_if_included_layout.itemAt(i).widget().setParent(None)



    def update_tag_completer(self):
        if self.tags_completer is None or self.skip_completer is None:
            print("Completers are not initialized yet.")
            return

        string_list_model = QStringListModel(self.tags_list)
        self.tags_completer.setModel(string_list_model)
        self.skip_completer.setModel(string_list_model)
        self.clear_tag_layouts()


    def valid_tags(self):
        return self.tags_list

    def add_tag_to_crop(self):
        tag = self.tags_input.text().strip()
        if tag and tag in self.valid_tags() and tag not in self.get_selected_tags_to_crop():
            tag_widget = QLabel(tag)
            tag_widget.setStyleSheet("background-color: #E0E0E0; border-radius: 10px; padding: 5px;")
            close_button = QPushButton("x")
            close_button.setFixedSize(20, 20)
            close_button.clicked.connect(lambda: self.remove_tag_from_crop(tag))
            layout = QHBoxLayout()
            layout.addWidget(tag_widget)
            layout.addWidget(close_button)
            layout.setContentsMargins(0, 0, 0, 0)
            widget = QWidget()
            widget.setLayout(layout)
            self.tags_to_crop_layout.addWidget(widget)
            self.tags_input.clear()            

    def remove_tag_from_crop(self, tag):
        for i in range(self.tags_to_crop_layout.count()):
            widget = self.tags_to_crop_layout.itemAt(i).widget()
            if widget:
                label = widget.findChild(QLabel)
                if label and label.text() == tag:
                    widget.setParent(None)
                    break

    def get_selected_tags_to_crop(self):
        tags = []
        for i in range(self.tags_to_crop_layout.count()):
            widget = self.tags_to_crop_layout.itemAt(i).widget()
            if widget:
                label = widget.findChild(QLabel)
                if label:
                    tags.append(label.text())
        return tags    

    def add_tag_to_skip(self):
        tag = self.skip_input.text().strip()
        if tag and tag in self.valid_tags() and tag not in self.get_selected_tags_to_skip():
            tag_widget = QLabel(tag)
            tag_widget.setStyleSheet("background-color: #E0E0E0; border-radius: 10px; padding: 5px;")
            close_button = QPushButton("x")
            close_button.setFixedSize(20, 20)
            close_button.clicked.connect(lambda: self.remove_tag_from_skip(tag))
            layout = QHBoxLayout()
            layout.addWidget(tag_widget)
            layout.addWidget(close_button)
            layout.setContentsMargins(0, 0, 0, 0)
            widget = QWidget()
            widget.setLayout(layout)
            self.skip_if_included_layout.addWidget(widget)
            self.skip_input.clear()

    def remove_tag_from_skip(self, tag):
        for i in range(self.skip_if_included_layout.count()):
            widget = self.skip_if_included_layout.itemAt(i).widget()
            if widget:
                label = widget.findChild(QLabel)
                if label and label.text() == tag:
                    widget.setParent(None)
                    break

    def get_selected_tags_to_skip(self):
        tags = []
        for i in range(self.skip_if_included_layout.count()):
            widget = self.skip_if_included_layout.itemAt(i).widget()
            if widget:
                label = widget.findChild(QLabel)
                if label:
                    tags.append(label.text())
        return tags

    def stop_processing(self):
        self.stop_processing_flag = True
        self.progress_bar.setValue(0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(self, 'model'):
            del self.model
        self.model = None
        gc.collect()

    def browse_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.source_folder_input.setText(folder)
            self.settings.setValue("source_folder", folder)

    def browse_export_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder:
            self.export_folder_input.setText(folder)
            self.settings.setValue("export_folder", folder)

    def browse_single_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Single Image", "", "Image Files (*.jpg *.png)")
        if file:
            self.single_image_input.setText(file)
            self.settings.setValue("single_image", file)

    def get_text_position(self, bbox, text_size):
        return (bbox[2] - text_size[0] - 2, bbox[1] + 2)

    def draw_text_with_outline(self, image, text, position, font, font_scale, text_color, outline_color, thickness):
        # Draw the text outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cv2.putText(image, text, (position[0] + dx, position[1] + dy), font, font_scale, outline_color, thickness * 3)
        # Draw the text in white
        cv2.putText(image, text, position, font, font_scale, text_color, thickness)

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        diagonal = np.sqrt(height**2 + width**2)
        font_scale = diagonal / 2000
        line_thickness = max(1, int(diagonal / 500))

        results = self.model.predict(source=image, save=False, verbose=False, device=self.device)

        object_tags = {}
        tag_boxes = {}
        class_colors = {}
        object_areas = []
        detected_tags = set()

        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                label = self.model.names[cls]
                bbox_xyxy = box.xyxy.cpu().numpy().astype(int).flatten()

                object_key = tuple(bbox_xyxy)

                if object_key not in object_tags:
                    object_tags[object_key] = set()
                object_tags[object_key].add(label)
                detected_tags.add(label)

                if self.mark_box:
                    area = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
                    object_areas.append(area)

        if self.mark_box:
            max_area = max(object_areas) if object_areas else 1
            min_area = min(object_areas) if object_areas else 1

            for bbox, tags in object_tags.items():
                for label in tags:
                    if label not in class_colors:
                        hue = np.random.rand()
                        saturation = 0.7 + 0.3 * np.random.rand()
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        lightness = 0.5 + 0.5 * ((area - min_area) / (max_area - min_area)) if max_area != min_area else 0.75

                        rgb_color = colorsys.hls_to_rgb(hue, lightness, saturation)
                        class_colors[label] = tuple(int(x * 255) for x in rgb_color)

                    color = class_colors[label]
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line_thickness)

                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)
                    text_pos = [bbox[2] - text_size[0] - 5, bbox[1] + text_size[1] + 5]

                    while any(np.linalg.norm(np.array(text_pos) - np.array(pos[:2])) < text_size[1] for pos in tag_boxes.values()):
                        text_pos[1] += text_size[1] + 5

                    text_pos[0] = max(0, min(text_pos[0], width - text_size[0]))
                    text_pos[1] = max(text_size[1], min(text_pos[1], height - 5))

                    self.draw_text_with_outline(image, label, tuple(text_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), (0, 0, 0), line_thickness)

                    tag_boxes[label] = bbox

        if self.ignore_tags:
            filename = os.path.basename(image_path)
            cv2.imwrite(os.path.join(self.export_folder, filename), image)
            return

        skip_tags = set(self.skip_if_included)
        if skip_tags and any(tag in detected_tags for tag in skip_tags):
            return  # Skip this image if it contains any skip tags

        if self.tags_to_crop:
            if len(self.tags_to_crop) == 1:
                # Single tag selection
                tag_to_crop = self.tags_to_crop[0]
                for bbox, tags in object_tags.items():
                    if tag_to_crop in tags:
                        cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        filename = os.path.basename(image_path)
                        cv2.imwrite(os.path.join(self.export_folder, filename), cropped_image)
                        return
            else:
                # Multiple tag selection
                all_tags = set().union(*object_tags.values())
                if all(tag in all_tags for tag in self.tags_to_crop):
                    # Find the bounding box that encompasses all selected tags
                    x1 = min(bbox[0] for bbox, tags in object_tags.items() if any(tag in tags for tag in self.tags_to_crop))
                    y1 = min(bbox[1] for bbox, tags in object_tags.items() if any(tag in tags for tag in self.tags_to_crop))
                    x2 = max(bbox[2] for bbox, tags in object_tags.items() if any(tag in tags for tag in self.tags_to_crop))
                    y2 = max(bbox[3] for bbox, tags in object_tags.items() if any(tag in tags for tag in self.tags_to_crop))

                    # Ensure bounding box is within image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    cropped_image = image[y1:y2, x1:x2]
                    filename = os.path.basename(image_path)
                    cv2.imwrite(os.path.join(self.export_folder, filename), cropped_image)
                    return
                else:
                    return  # Not all tags found
        else:
            if object_tags and not self.ignore_tags and self.mark_box and self.tags_to_crop:
                largest_box = max(object_tags.keys(), key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                # Ensure bounding box is within image boundaries
                x1, y1, x2, y2 = largest_box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                cropped_image = image[y1:y2, x1:x2]
            else:
                cropped_image = image  # Use the full image if no objects were detected or ignore_tags is enabled

            filename = os.path.basename(image_path)
            cv2.imwrite(os.path.join(self.export_folder, filename), cropped_image)

        QApplication.processEvents()

    def process_images(self):
        self.model_path = self.model_path_input.currentText()
        self.device = self.device_combo.currentText()
        self.source_folder = self.source_folder_input.text()
        self.export_folder = self.export_folder_input.text()
        self.single_image_path = self.single_image_input.text()
        self.tags_to_crop = self.get_selected_tags_to_crop()
        self.skip_if_included = self.get_selected_tags_to_skip()
        self.mark_box = self.mark_box_checkbox.isChecked()
        self.ignore_tags = self.ignore_tags_checkbox.isChecked()


        self.stop_button.setEnabled(True)
        self.process_button.setEnabled(False)

        # Load the YOLO model
        try:
            if self.model_path.endswith('.engine'):
                self.model = YOLO(self.model_path)
                print(f"Loaded TensorRT engine: {self.model_path}")
            else:
                self.model = YOLO(self.model_path)
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                    print(f"Using CUDA ({self.device}) for acceleration")
                else:
                    print("CUDA not available or not specified, using CPU")
            
            print("Model classes:", self.model.names)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.stop_button.setEnabled(False)
            self.process_button.setEnabled(True)
            return

        # Create the export folder if it doesn't exist
        if not os.path.exists(self.export_folder):
            os.makedirs(self.export_folder)

        # Check if single image is selected
        if os.path.exists(self.single_image_path):
            print("Processing single image...")
            self.process_image(self.single_image_path)
        else:
            print("Processing images in the source folder...")
            image_files = [filename for filename in os.listdir(self.source_folder) if filename.endswith((".jpg", ".png", ".jpeg"))]
            num_images = len(image_files)
            processed_images = 0
            for index, filename in enumerate(image_files):
                if self.stop_processing_flag:
                    break
                image_path = os.path.join(self.source_folder, filename)
                self.process_image(image_path)
                processed_images += 1
                progress = processed_images / num_images * 100
                self.progress_bar.setValue(int(progress))
                QApplication.processEvents()

        self.stop_button.setEnabled(False)
        self.process_button.setEnabled(True)
        self.stop_processing_flag = False
        self.progress_bar.setValue(0)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()    
        QMessageBox.information(self, "Processing Complete", "Image processing completed!")

    def closeEvent(self, event: QCloseEvent) -> None:
        # Perform any cleanup if necessary
        self.stop_processing_flag = True  # Stop any ongoing processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(self, 'model'):
            del self.model
        self.model = None
        gc.collect()
        
        event.accept()
        QApplication.instance().quit()


if __name__ == "__main__":
    app = QApplication([])
    window = CropGUI()
    window.show()
    app.exec_()

