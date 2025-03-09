from ultralytics import YOLO
import os
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


class YOLOv11Processor:
    def __init__(self, model_path: str, confidence_threshold=0.25, iou_threshold=0.25):
        """
        Initializes the YOLOv11 model for automatic labeling.

        :param model_path: Path to the trained YOLOv11 model file (e.g., 'yolov11.pt').
        :param confidence_threshold: Confidence threshold for detection.
        :param iou_threshold: IOU threshold for non-maximum suppression.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=self.confidence_threshold,
            device="0",
        )

    def predict(self, image_path: str):
        """
        Runs the YOLOv11 model on a single image and returns the parsed detections.

        :param image_path: Path to the input image.
        :return: List of detected objects with class, confidence, and bounding box.
        """
        # detections = self.model.predict(
        #     source=image_path, conf=self.confidence_threshold, iou=self.iou_threshold
        # )

        detections = get_sliced_prediction(
            image_path,
            self.model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        return self._parse_detections(detections)

    def label_images(self, image_path: str, dataset_path: str, extension=".jpg", save_results=True):
        """
        Runs the YOLOv11 model to label images.

        :param image_path: Path where input images are stored.
        :param dataset_path: Path where labeled dataset should be saved.
        :param extension: Image file extension (default is .jpg).
        :param save_results: Whether to save the labeled dataset.
        """
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        image_files = [f for f in os.listdir(image_path) if f.endswith(extension)]
        results = []

        for image_file in image_files:
            img_full_path = os.path.join(image_path, image_file)
            parsed_detections = self.predict(img_full_path)  # Use predict() which already parses detections

            result_data = {"image": image_file, "detections": parsed_detections}
            results.append(result_data)

            if save_results:
                output_path = os.path.join(dataset_path, image_file.replace(extension, "_labeled.txt"))
                with open(output_path, "w") as f:
                    for det in parsed_detections:
                        f.write(f"{det['class']} {det['confidence']} {det['bbox']}\n")

        return results

    def _parse_detections(self, detections):
        """
        Parses YOLOv11 model detections into a structured format.

        :param detections: Model predictions.
        :return: List of dictionaries with class label, confidence, and bounding box coordinates.
        """
        # ✅ Get class names from SAHI model
        class_names = self.model.category_mapping  # Dictionary {id: "class_name"}

        parsed_detections = []

        for detection in detections.object_prediction_list:
            confidence = float(detection.score.value)

            if confidence < self.confidence_threshold:  # Ignore low-confidence detections
                continue

            class_id = int(detection.category.id)
            class_label = class_names.get(class_id, "unknown")  # Get class label safely
            x_min, y_min, x_max, y_max = detection.bbox.to_xyxy()  # Convert bounding box format

            parsed_detections.append({
                "class": class_label,  # Class label instead of class number
                "confidence": confidence,
                "bbox": [x_min, y_min, x_max, y_max]  # Bounding box format
            })

        return parsed_detections

