from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology


class GroundingDINOProcessor:
    def __init__(self, ontology_dict: dict, image_path: str, dataset_path: str):
        """
        Initializes the GroundingDINO model for automatic labeling.

        :param ontology_dict: Dictionary defining the ontology mapping for detection.
        :param image_path: Path where input images are stored.
        :param dataset_path: Path where labeled dataset should be saved.
        """
        self.image_path = image_path
        self.dataset_path = dataset_path
        self.ontology = CaptionOntology(ontology_dict)
        self.model = GroundingDINO(ontology=self.ontology, box_threshold=0.25, text_threshold=0.25)

    def label_images(self, extension=".jpg", sahi=True, human_in_the_loop=True, nms_settings='class_specific'):
        """
        Runs the GroundingDINO model to label images.

        :param extension: Image file extension (default is .jpg).
        :param sahi: Whether to use SAHI for small object detection.
        :param human_in_the_loop: Whether to include human verification.
        :param nms_settings: Non-maximum suppression settings.
        """
        dataset = self.model.label(
            input_folder=self.image_path,
            extension=extension,
            output_folder=self.dataset_path,
            sahi=sahi,
            # human_in_the_loop=human_in_the_loop,
            nms_settings=nms_settings
        )
        return dataset  