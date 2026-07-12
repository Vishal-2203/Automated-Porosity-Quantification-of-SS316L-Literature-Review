"""
===============================================================================
POROSITY ANALYZER
Professional Metallography & Porosity Analysis Software

Single File Edition

Author  : ChatGPT + User
Version : 2.0
===============================================================================
"""

###############################################################################
# Imports
###############################################################################

import sys
import os
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import (
    Qt,
    QRectF,
    Signal,
    QAbstractTableModel,
    QModelIndex,
)

from PySide6.QtGui import (
    QAction,
    QColor,
    QImage,
    QPixmap,
)

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QGroupBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QStatusBar,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QCheckBox,
    QTableWidget,
    QTableView,
    QAbstractItemView,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
)

###############################################################################
# Application Information
###############################################################################

APP_NAME = "Porosity Analyzer"

VERSION = "2.0"

###############################################################################
# Logging
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(APP_NAME)

###############################################################################
# Configuration
###############################################################################


class AppConfig:
    """
    Stores every configurable parameter used
    throughout the application.
    """

    def __init__(self):

        # ----------------------------------------------------------
        # Window
        # ----------------------------------------------------------

        self.window_width = 1600
        self.window_height = 900

        # ----------------------------------------------------------
        # Viewer
        # ----------------------------------------------------------

        self.dark_theme = True

        self.show_grid = False

        self.show_scale_bar = False

        # ----------------------------------------------------------
        # CLAHE
        # ----------------------------------------------------------

        self.clahe_clip = 2.0

        self.clahe_tile = 8

        # ----------------------------------------------------------
        # Threshold
        # ----------------------------------------------------------

        self.threshold_method = "Adaptive Gaussian"

        self.fixed_threshold = 127

        self.block_size = 31

        self.constant_c = 5

        self.detect_dark_objects = True

        self.blur_kernel = 3

        self.window_size = 31

        self.k = 0.20

        self.r = 128

        # ----------------------------------------------------------
        # Morphology
        # ----------------------------------------------------------

        self.kernel_shape = "Ellipse"

        self.kernel_size = 3

        self.iterations = 1

        self.opening = False

        self.closing = False

        self.erosion = False

        self.dilation = False

        # ----------------------------------------------------------
        # Connected Components
        # ----------------------------------------------------------

        self.minimum_area = 1

        self.maximum_area = 1000000

        # ----------------------------------------------------------
        # Histogram
        # ----------------------------------------------------------

        self.histogram_bins = 50

        # ----------------------------------------------------------
        # Calibration
        # ----------------------------------------------------------

        self.pixel_size = 1.0

        self.calibration_name = "Custom"

###############################################################################
# End of Part 1
###############################################################################
###############################################################################
# Processing Pipeline
###############################################################################


class ProcessingPipeline:
    """
    Stores every intermediate processing stage.

    Original
        â†“
    Gray
        â†“
    CLAHE
        â†“
    Binary
        â†“
    Morphology
        â†“
    Labels
        â†“
    Overlay
    """

    def __init__(self):

        self.clear()

    # ----------------------------------------------------------------------

    def clear(self):

        self.images = {
            "original": None,
            "gray": None,
            "clahe": None,
            "binary": None,
            "morphology": None,
            "labels": None,
            "overlay": None,

            # Internal caches
            "label_matrix": None,
            "component_stats": None,
            "component_centroids": None,
        }

        self.measurements = None

        self.statistics = None

    # ----------------------------------------------------------------------

    def get(self, stage):

        return self.images.get(stage)

    # ----------------------------------------------------------------------

    def set(self, stage, image):

        self.images[stage] = image

    # ----------------------------------------------------------------------

    def exists(self, stage):

        return self.images.get(stage) is not None

    # ----------------------------------------------------------------------

    def invalidate_after(self, stage):

        order = [

            "original",

            "gray",

            "clahe",

            "binary",

            "morphology",

            "labels",

            "overlay",

        ]

        if stage not in order:

            return

        index = order.index(stage)

        for item in order[index + 1:]:

            self.images[item] = None

        self.measurements = None

        self.statistics = None

    # ----------------------------------------------------------------------

    def stage_names(self):

        return list(self.images.keys())


###############################################################################
# Image Conversion Utilities
###############################################################################


def cv_to_qimage(image):

    """
    Convert OpenCV image to Qt image.
    """

    if image is None:

        return None

    # ----------------------------------------------------------
    # Grayscale
    # ----------------------------------------------------------

    if len(image.shape) == 2:

        h, w = image.shape

        return QImage(

            image.data,

            w,

            h,

            image.strides[0],

            QImage.Format_Grayscale8,

        ).copy()

    # ----------------------------------------------------------
    # Color
    # ----------------------------------------------------------

    rgb = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )

    h, w = rgb.shape[:2]

    return QImage(

        rgb.data,

        w,

        h,

        rgb.strides[0],

        QImage.Format_RGB888,

    ).copy()


# --------------------------------------------------------------------------


def cv_to_pixmap(image):

    qimage = cv_to_qimage(image)

    if qimage is None:

        return QPixmap()

    return QPixmap.fromImage(qimage)


###############################################################################
# Utility Functions
###############################################################################


def ensure_odd(value):

    """
    Many OpenCV filters require odd kernel sizes.
    """

    if value % 2 == 0:

        value += 1

    return value


# --------------------------------------------------------------------------


def create_kernel(shape, size):

    """
    Returns the selected morphology kernel.
    """

    size = ensure_odd(size)

    if shape == "Rectangle":

        return cv2.getStructuringElement(

            cv2.MORPH_RECT,

            (size, size)

        )

    elif shape == "Cross":

        return cv2.getStructuringElement(

            cv2.MORPH_CROSS,

            (size, size)

        )

    return cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (size, size)

    )


# --------------------------------------------------------------------------


def image_information(image):

    """
    Returns image metadata.
    """

    if image is None:

        return {}

    h, w = image.shape[:2]

    if len(image.shape) == 2:

        channels = 1

    else:

        channels = image.shape[2]

    return {

        "Width": w,

        "Height": h,

        "Channels": channels,

        "Type": str(image.dtype),

    }


# --------------------------------------------------------------------------


def normalize_binary(image):

    """
    Ensures binary image contains only

    0

    and

    255
    """

    return (

        image > 0

    ).astype(

        np.uint8

    ) * 255


###############################################################################
# End of Part 2
###############################################################################
###############################################################################
# Image Viewer
###############################################################################


class ImageViewer(QGraphicsView):
    """
    Professional image viewer.

    Features
    --------
    â€¢ Mouse wheel zoom
    â€¢ Drag to pan
    â€¢ Fit to window
    â€¢ Reset zoom
    â€¢ Supports grayscale and RGB images
    â€¢ Future overlay support
    """

    def __init__(self):

        super().__init__()

        self.scene = QGraphicsScene(self)

        self.setScene(self.scene)

        self.image_item = QGraphicsPixmapItem()

        self.scene.addItem(self.image_item)

        self.current_image = None

        self.zoom_level = 0

        self.zoom_factor = 1.25

        self.min_zoom = -30

        self.max_zoom = 40

        self.initialize_view()

    # ------------------------------------------------------------------

    def initialize_view(self):

        self.setBackgroundBrush(QColor(35, 35, 35))

        self.setFrameShape(QGraphicsView.NoFrame)

        self.setDragMode(

            QGraphicsView.ScrollHandDrag

        )

        self.setTransformationAnchor(

            QGraphicsView.AnchorUnderMouse

        )

        self.setResizeAnchor(

            QGraphicsView.AnchorUnderMouse

        )

        self.setViewportUpdateMode(

            QGraphicsView.FullViewportUpdate

        )

        self.setHorizontalScrollBarPolicy(

            Qt.ScrollBarAsNeeded

        )

        self.setVerticalScrollBarPolicy(

            Qt.ScrollBarAsNeeded

        )

    # ------------------------------------------------------------------

    def clear(self):

        self.current_image = None

        self.image_item.setPixmap(

            QPixmap()

        )

        self.scene.clear()

        self.image_item = QGraphicsPixmapItem()

        self.scene.addItem(

            self.image_item

        )

        self.zoom_level = 0

    # ------------------------------------------------------------------

    def has_image(self):

        return not self.image_item.pixmap().isNull()

    # ------------------------------------------------------------------

    def set_image(self, image):

        if image is None:

            self.clear()

            return

        self.current_image = image.copy()

        pixmap = cv_to_pixmap(image)

        self.image_item.setPixmap(

            pixmap

        )

        self.scene.setSceneRect(

            QRectF(

                pixmap.rect()

            )

        )

        self.fit_to_window()

    # ------------------------------------------------------------------

    def current(self):

        return self.current_image

    # ------------------------------------------------------------------

    def fit_to_window(self):

        if not self.has_image():

            return

        self.fitInView(

            self.scene.sceneRect(),

            Qt.KeepAspectRatio

        )

        self.zoom_level = 0

    # ------------------------------------------------------------------

    def reset_zoom(self):

        self.fit_to_window()

    # ------------------------------------------------------------------

    def actual_size(self):

        if not self.has_image():

            return

        self.resetTransform()

        self.zoom_level = 0

    # ------------------------------------------------------------------

    def zoom_in(self):

        if not self.has_image():

            return

        if self.zoom_level >= self.max_zoom:

            return

        self.scale(

            self.zoom_factor,

            self.zoom_factor

        )

        self.zoom_level += 1

    # ------------------------------------------------------------------

    def zoom_out(self):

        if not self.has_image():

            return

        if self.zoom_level <= self.min_zoom:

            return

        self.scale(

            1 / self.zoom_factor,

            1 / self.zoom_factor

        )

        self.zoom_level -= 1

    # ------------------------------------------------------------------

    def wheelEvent(self, event):

        if not self.has_image():

            return

        if event.angleDelta().y() > 0:

            self.zoom_in()

        else:

            self.zoom_out()

    # ------------------------------------------------------------------

    def resizeEvent(self, event):

        super().resizeEvent(event)

        if self.zoom_level == 0:

            self.fit_to_window()

    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, event):

        if event.button() == Qt.LeftButton:

            self.fit_to_window()

        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------------

    def center_image(self):

        if not self.has_image():

            return

        self.centerOn(

            self.image_item

        )

    # ------------------------------------------------------------------

    def image_size(self):

        if self.current_image is None:

            return None

        h, w = self.current_image.shape[:2]

        return (

            w,

            h

        )

    # ------------------------------------------------------------------

    def refresh(self):

        if self.current_image is None:

            return

        self.set_image(

            self.current_image

        )

    # ------------------------------------------------------------------

    def set_background(self, color):

        self.setBackgroundBrush(

            color

        )

    # ------------------------------------------------------------------

    def show_placeholder(self):

        self.clear()

        text = self.scene.addText(

            "Open an image to begin"

        )

        font = text.font()

        font.setPointSize(18)

        font.setBold(True)

        text.setFont(font)

        text.setDefaultTextColor(

            QColor(190, 190, 190)

        )

        rect = text.boundingRect()

        text.setPos(

            -rect.width() / 2,

            -rect.height() / 2

        )

        self.scene.setSceneRect(

            -400,

            -300,

            800,

            600

        )

        self.centerOn(text)

    # ------------------------------------------------------------------

    def overlay_supported(self):

        return True

    # ------------------------------------------------------------------

    def mousePressEvent(self, event):

        super().mousePressEvent(event)

    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):

        super().mouseMoveEvent(event)

    # ------------------------------------------------------------------

    def mouseReleaseEvent(self, event):

        super().mouseReleaseEvent(event)
###############################################################################
# Image Processor
###############################################################################

from skimage.filters import (
    threshold_otsu,
    threshold_multiotsu,
    threshold_triangle,
    threshold_yen,
    threshold_li,
    threshold_sauvola,
    threshold_niblack,
)


class ImageProcessor:
    """
    Performs every image processing operation.

    GUI never directly calls OpenCV.
    """

    def __init__(self, pipeline, config):

        self.pipeline = pipeline
        self.config = config

    # ------------------------------------------------------------------

    def clear(self):

        self.pipeline.clear()

    # ------------------------------------------------------------------

    def load_image(self, filename):

        image = cv2.imread(filename)

        if image is None:
            raise RuntimeError(
                f"Unable to open\n{filename}"
            )

        self.set_original(image)

        return image

    # ------------------------------------------------------------------

    def set_original(self, image):

        self.pipeline.clear()

        self.pipeline.set(

            "original",

            image

        )

        return image

    # ------------------------------------------------------------------

    def original(self):

        return self.pipeline.get(

            "original"

        )

    # ------------------------------------------------------------------

    def image_information(self):

        return image_information(

            self.original()

        )

    # ------------------------------------------------------------------

    def width(self):

        image = self.original()

        if image is None:
            return 0

        return image.shape[1]

    # ------------------------------------------------------------------

    def height(self):

        image = self.original()

        if image is None:
            return 0

        return image.shape[0]

    # ------------------------------------------------------------------

    def gray(self):

        if self.pipeline.exists(

            "gray"

        ):

            return self.pipeline.get(

                "gray"

            )

        image = self.original()

        if image is None:
            return None

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )

        self.pipeline.set(

            "gray",

            gray

        )

        return gray

    # ------------------------------------------------------------------

    def clahe(self):

        if self.pipeline.exists(

            "clahe"

        ):

            return self.pipeline.get(

                "clahe"

            )

        gray = self.gray()

        if gray is None:

            return None

        clahe = cv2.createCLAHE(

            clipLimit=self.config.clahe_clip,

            tileGridSize=(

                self.config.clahe_tile,

                self.config.clahe_tile

            )

        )

        result = clahe.apply(

            gray

        )

        self.pipeline.set(

            "clahe",

            result

        )

        return result

    # ------------------------------------------------------------------

    def current_gray_source(self):

        """
        Thresholding always starts from CLAHE.
        """

        gray = self.clahe()

        if gray is None:
            return None

        if self.config.blur_kernel > 1:
            gray = cv2.GaussianBlur(
                gray,
                (self.config.blur_kernel, self.config.blur_kernel),
                0
            )

        return gray

    # ------------------------------------------------------------------

    def image(self, stage):

        stage = stage.lower()

        if stage == "original":

            return self.original()

        elif stage == "gray":

            return self.gray()

        elif stage == "clahe":

            return self.clahe()

        elif stage == "binary":

            return self.binary()

        elif stage == "morphology":

            return self.morphology()

        elif stage == "overlay":

            return self.overlay()

        return self.pipeline.get(stage)
        # ------------------------------------------------------------------
    # Binary Image
    # ------------------------------------------------------------------

    def binary(self):

        """
        Returns the binary image according to the
        currently selected threshold algorithm.
        """

        if self.pipeline.exists("binary"):

            return self.pipeline.get("binary")

        method = self.config.threshold_method

        if method == "Fixed":
            return self.threshold_fixed()

        elif method == "Otsu":
            return self.threshold_otsu()

        elif method == "Multi-Otsu":
            return self.threshold_multiotsu()

        elif method == "Triangle":
            return self.threshold_triangle()

        elif method == "Yen":
            return self.threshold_yen()

        elif method == "Li":
            return self.threshold_li()

        elif method == "Sauvola":
            return self.threshold_sauvola()

        elif method == "Niblack":
            return self.threshold_niblack()

        elif method == "Adaptive Mean":
            return self.threshold_adaptive_mean()

        elif method == "Adaptive Gaussian":
            return self.threshold_adaptive_gaussian()

        return self.threshold_otsu()

    # ------------------------------------------------------------------

    def save_binary(self, image):

        image = normalize_binary(image)

        self.pipeline.set(

            "binary",

            image

        )

        return image

    # ------------------------------------------------------------------
    # Fixed Threshold
    # ------------------------------------------------------------------

    def threshold_fixed(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        threshold_type = cv2.THRESH_BINARY_INV if self.config.detect_dark_objects else cv2.THRESH_BINARY

        _, binary = cv2.threshold(

            gray,

            self.config.fixed_threshold,

            255,

            threshold_type

        )

        return self.save_binary(binary)

    # ------------------------------------------------------------------
    # Otsu
    # ------------------------------------------------------------------

    def threshold_otsu(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        threshold = threshold_otsu(gray)

        if self.config.detect_dark_objects:
            binary = (gray < threshold).astype(np.uint8) * 255
        else:
            binary = (gray > threshold).astype(np.uint8) * 255

        return self.save_binary(binary)

    # ------------------------------------------------------------------
    # Multi-Otsu
    # ------------------------------------------------------------------

    def threshold_multiotsu(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        thresholds = threshold_multiotsu(

            gray,

            classes=3

        )

        regions = np.digitize(

            gray,

            bins=thresholds

        )

        if self.config.detect_dark_objects:
            binary = (regions < 2).astype(np.uint8) * 255
        else:
            binary = (regions == 2).astype(np.uint8) * 255

        return self.save_binary(binary)

    # ------------------------------------------------------------------
    # Triangle
    # ------------------------------------------------------------------

    def threshold_triangle(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        t = threshold_triangle(gray)

        if self.config.detect_dark_objects:
            binary = (gray < t).astype(np.uint8) * 255
        else:
            binary = (gray > t).astype(np.uint8) * 255

        return self.save_binary(binary)

    # ------------------------------------------------------------------
    # Yen
    # ------------------------------------------------------------------

    def threshold_yen(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        t = threshold_yen(gray)

        if self.config.detect_dark_objects:
            binary = (gray < t).astype(np.uint8) * 255
        else:
            binary = (gray > t).astype(np.uint8) * 255

        return self.save_binary(binary)

    # ------------------------------------------------------------------
    # Li
    # ------------------------------------------------------------------

    def threshold_li(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        t = threshold_li(gray)

        if self.config.detect_dark_objects:
            binary = (gray < t).astype(np.uint8) * 255
        else:
            binary = (gray > t).astype(np.uint8) * 255

        return self.save_binary(binary)
    # ------------------------------------------------------------------
    # Sauvola
    # ------------------------------------------------------------------

    def threshold_sauvola(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        window = ensure_odd(

            self.config.window_size

        )

        threshold = threshold_sauvola(

            gray,

            window_size=window,

            k=self.config.k,

            r=self.config.r

        )

        if self.config.detect_dark_objects:
            binary = (gray < threshold).astype(np.uint8) * 255
        else:
            binary = (gray > threshold).astype(np.uint8) * 255

        return self.save_binary(binary)

    # ------------------------------------------------------------------
    # Niblack
    # ------------------------------------------------------------------

    def threshold_niblack(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        window = ensure_odd(

            self.config.window_size

        )

        threshold = threshold_niblack(

            gray,

            window_size=window,

            k=self.config.k

        )

        if self.config.detect_dark_objects:
            binary = (gray < threshold).astype(np.uint8) * 255
        else:
            binary = (gray > threshold).astype(np.uint8) * 255

        return self.save_binary(binary)

    # ------------------------------------------------------------------
    # Adaptive Mean
    # ------------------------------------------------------------------

    def threshold_adaptive_mean(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        block = ensure_odd(

            self.config.block_size

        )

        threshold_type = cv2.THRESH_BINARY_INV if self.config.detect_dark_objects else cv2.THRESH_BINARY

        binary = cv2.adaptiveThreshold(

            gray,

            255,

            cv2.ADAPTIVE_THRESH_MEAN_C,

            threshold_type,

            block,

            self.config.constant_c

        )

        return self.save_binary(binary)

    # ------------------------------------------------------------------
    # Adaptive Gaussian
    # ------------------------------------------------------------------

    def threshold_adaptive_gaussian(self):

        gray = self.current_gray_source()

        if gray is None:
            return None

        block = ensure_odd(

            self.config.block_size

        )

        threshold_type = cv2.THRESH_BINARY_INV if self.config.detect_dark_objects else cv2.THRESH_BINARY

        binary = cv2.adaptiveThreshold(

            gray,

            255,

            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            threshold_type,

            block,

            self.config.constant_c

        )

        return self.save_binary(binary)

    ####################################################################
    # Morphology
    ####################################################################

    def morphology(self):

        """
        Performs all selected morphology operations.
        """

        if self.pipeline.exists("morphology"):

            return self.pipeline.get("morphology")

        image = self.binary()

        kernel = create_kernel(

            self.config.kernel_shape,

            self.config.kernel_size

        )

        result = image.copy()

        if self.config.opening:

            result = cv2.morphologyEx(

                result,

                cv2.MORPH_OPEN,

                kernel,

                iterations=self.config.iterations

            )

        if self.config.closing:

            result = cv2.morphologyEx(

                result,

                cv2.MORPH_CLOSE,

                kernel,

                iterations=self.config.iterations

            )

        if self.config.erosion:

            result = cv2.erode(

                result,

                kernel,

                iterations=self.config.iterations

            )

        if self.config.dilation:

            result = cv2.dilate(

                result,

                kernel,

                iterations=self.config.iterations

            )

        self.pipeline.set(

            "morphology",

            result

        )

        return result
        ####################################################################
    # Connected Components
    ####################################################################

    def labels(self):

        """
        Labels every connected pore.
        """

        if self.pipeline.exists("labels"):

            return self.pipeline.get("labels")

        binary = self.morphology()

        num_labels, label_image, stats, centroids = cv2.connectedComponentsWithStats(

            binary,

            connectivity=8

        )

        self.pipeline.set(

            "label_matrix",

            label_image

        )

        self.pipeline.set(

            "component_stats",

            stats

        )

        self.pipeline.set(

            "component_centroids",

            centroids

        )

        color = np.zeros(

            (label_image.shape[0],

             label_image.shape[1],

             3),

            dtype=np.uint8

        )

        rng = np.random.default_rng(12345)

        for label in range(1, num_labels):

            area = stats[label, cv2.CC_STAT_AREA]

            if area < self.config.minimum_area:

                continue

            if area > self.config.maximum_area:

                continue

            rgb = rng.integers(

                50,

                255,

                size=3,

                dtype=np.uint8

            )

            color[label_image == label] = rgb

        self.pipeline.set(

            "labels",

            color

        )

        return color

    # ------------------------------------------------------------------

    def overlay(self):

        """
        Draw colored pores on top of the original image.
        """

        if self.pipeline.exists("overlay"):

            return self.pipeline.get("overlay")

        original = self.original()

        labels = self.labels()

        if original is None:

            return None

        overlay = cv2.addWeighted(

            original,

            0.70,

            labels,

            0.30,

            0

        )

        self.pipeline.set(

            "overlay",

            overlay

        )

        return overlay

    # ------------------------------------------------------------------

    def number_of_components(self):

        stats = self.pipeline.get(

            "component_stats"

        )

        if stats is None:

            self.labels()

            stats = self.pipeline.get(

                "component_stats"

            )

        if stats is None:

            return 0

        count = 0

        for label in range(1, len(stats)):

            area = stats[label, cv2.CC_STAT_AREA]

            if area < self.config.minimum_area:

                continue

            if area > self.config.maximum_area:

                continue

            count += 1

        return count

    # ------------------------------------------------------------------

    def component_mask(self, label):

        matrix = self.pipeline.get(

            "label_matrix"

        )

        if matrix is None:

            self.labels()

            matrix = self.pipeline.get(

                "label_matrix"

            )

        if matrix is None:

            return None

        return (

            matrix == label

        ).astype(

            np.uint8

        ) * 255

    # ------------------------------------------------------------------

    def component_statistics(self):

        """
        Returns raw connected component data.
        """

        if self.pipeline.get("component_stats") is None:

            self.labels()

        return {

            "labels": self.pipeline.get(

                "label_matrix"

            ),

            "stats": self.pipeline.get(

                "component_stats"

            ),

            "centroids": self.pipeline.get(

                "component_centroids"

            )

        }
    ####################################################################
    # Measurements
    ####################################################################

    def measurements(self):

        """
        Calculates pore measurements.

        Returns
        -------
        pandas.DataFrame
        """

        if self.pipeline.measurements is not None:

            return self.pipeline.measurements

        labels = self.pipeline.get("label_matrix")

        stats = self.pipeline.get("component_stats")

        if labels is None or stats is None:

            self.labels()

            labels = self.pipeline.get("label_matrix")

            stats = self.pipeline.get("component_stats")

        rows = []

        pixel = self.config.pixel_size

        num_labels = labels.max()

        for label in range(1, num_labels + 1):

            component_area = stats[label, cv2.CC_STAT_AREA]

            if component_area < self.config.minimum_area:
                continue

            if component_area > self.config.maximum_area:
                continue

            mask = (labels == label).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )

            if len(contours) == 0:
                continue

            contour = max(
                contours,
                key=cv2.contourArea
            )

            perimeter = cv2.arcLength(
                contour,
                True
            )

            contour_area_px = cv2.contourArea(
                contour
            )

            area_px = float(component_area)

            equivalent_diameter = np.sqrt(
                4.0 * area_px / np.pi
            )

            circularity = (
                4.0 * np.pi * area_px
                / (perimeter * perimeter)
                if perimeter > 0
                else 0
            )

            hull = cv2.convexHull(
                contour
            )

            hull_area = cv2.contourArea(
                hull
            )

            solidity = (
                min(contour_area_px, area_px) / hull_area
                if hull_area > 0
                else 1.0
            )

            x, y, w, h = cv2.boundingRect(
                contour
            )

            aspect_ratio = (
                w / h
                if h > 0
                else 0
            )

            centroids = self.pipeline.get(
                "component_centroids"
            )

            centroid = centroids[label]

            major_axis = 0
            minor_axis = 0
            orientation = 0

            if len(contour) >= 5:

                ellipse = cv2.fitEllipse(
                    contour
                )

                (_, _), (a, b), angle = ellipse

                major_axis = max(a, b)

                minor_axis = min(a, b)

                orientation = angle

            rows.append({

                "ID": label,

                "Area (px²)": area_px,

                "Area (µm²)": area_px * pixel * pixel,

                "Perimeter (px)": perimeter,

                "Perimeter (µm)": perimeter * pixel,

                "Equivalent Diameter (px)": equivalent_diameter,

                "Equivalent Diameter (µm)": equivalent_diameter * pixel,

                "Circularity": circularity,

                "Solidity": solidity,

                "Aspect Ratio": aspect_ratio,

                "Major Axis": major_axis,

                "Minor Axis": minor_axis,

                "Orientation": orientation,

                "Centroid X": centroid[0],

                "Centroid Y": centroid[1],

                "Bounding X": x,

                "Bounding Y": y,

                "Bounding Width": w,

                "Bounding Height": h,

            })

        df = pd.DataFrame(rows)

        self.pipeline.measurements = df

        return df

    ####################################################################
    # Statistics
    ####################################################################

    def statistics(self):

        """
        Calculates global statistics.
        """

        if self.pipeline.statistics is not None:

            return self.pipeline.statistics

        df = self.measurements()

        if df.empty:

            self.pipeline.statistics = {}

            return {}

        image = self.original()

        image_area = (
            image.shape[0] *
            image.shape[1]
        )

        pixel_area = self.config.pixel_size * self.config.pixel_size

        pore_area = df[
            "Area (px²)"
        ].sum()

        pore_area_um2 = pore_area * pixel_area

        image_area_um2 = image_area * pixel_area

        stats = {

            "Number of Pores":
                len(df),

            "Porosity (%)":
                100.0 * pore_area / image_area,

            "Mean Area":
                df["Area (px²)"].mean(),

            "Median Area":
                df["Area (px²)"].median(),

            "Std Area":
                df["Area (px²)"].std(),

            "Largest Pore":
                df["Area (px²)"].max(),

            "Smallest Pore":
                df["Area (px²)"].min(),

            "Mean Diameter":
                df[
                    "Equivalent Diameter (px)"
                ].mean(),

            "Density":
                len(df) / image_area,

            "Density (per µm²)":
                len(df) / image_area_um2 if image_area_um2 > 0 else 0,

            "Pore Area (µm²)":
                pore_area_um2,

        }

        self.pipeline.statistics = stats

        return stats
###############################################################################
# CLAHE Panel
###############################################################################


class CLAHEPanel(QGroupBox):

    parametersChanged = Signal()

    def __init__(self, config):

        super().__init__("CLAHE")

        self.config = config

        self.create_widgets()

        self.create_layout()

        self.connect_signals()

    # ------------------------------------------------------------------

    def create_widgets(self):

        self.clip = QDoubleSpinBox()

        self.clip.setRange(

            0.1,

            20.0

        )

        self.clip.setSingleStep(

            0.1

        )

        self.clip.setValue(

            self.config.clahe_clip

        )

        self.tile = QSpinBox()

        self.tile.setRange(

            2,

            64

        )

        self.tile.setSingleStep(

            2

        )

        self.tile.setValue(

            self.config.clahe_tile

        )

    # ------------------------------------------------------------------

    def create_layout(self):

        layout = QFormLayout()

        layout.addRow(

            "Clip Limit",

            self.clip

        )

        layout.addRow(

            "Tile Size",

            self.tile

        )

        self.setLayout(layout)

    # ------------------------------------------------------------------

    def connect_signals(self):

        self.clip.valueChanged.connect(

            self.changed

        )

        self.tile.valueChanged.connect(

            self.changed

        )

    # ------------------------------------------------------------------

    def changed(self):

        self.config.clahe_clip = self.clip.value()

        self.config.clahe_tile = self.tile.value()

        self.parametersChanged.emit()
###############################################################################
# Threshold Panel
###############################################################################


class ThresholdPanel(QGroupBox):

    parametersChanged = Signal()

    def __init__(self, config):

        super().__init__("Threshold")

        self.config = config

        self.create_widgets()

        self.create_layout()

        self.connect_signals()

        self.update_ui()

    # ------------------------------------------------------------------

    def create_widgets(self):

        self.method = QComboBox()

        self.method.addItems([

            "Fixed",

            "Otsu",

            "Multi-Otsu",

            "Triangle",

            "Yen",

            "Li",

            "Sauvola",

            "Niblack",

            "Adaptive Mean",

            "Adaptive Gaussian"

        ])

        self.method.setCurrentText(

            self.config.threshold_method

        )

        self.threshold = QSlider(

            Qt.Horizontal

        )

        self.threshold.setRange(

            0,

            255

        )

        self.threshold.setValue(

            self.config.fixed_threshold

        )

        self.value = QLabel(

            str(self.config.fixed_threshold)

        )

        self.dark_objects = QCheckBox("Detect Dark Objects")

        self.dark_objects.setChecked(

            self.config.detect_dark_objects

        )

        self.block_size = QSpinBox()

        self.block_size.setRange(

            3,

            101

        )

        self.block_size.setSingleStep(

            2

        )

        self.block_size.setValue(

            ensure_odd(self.config.block_size)

        )

        self.constant_c = QSpinBox()

        self.constant_c.setRange(

            -50,

            50

        )

        self.constant_c.setValue(

            self.config.constant_c

        )

        self.window_size = QSpinBox()

        self.window_size.setRange(

            3,

            101

        )

        self.window_size.setSingleStep(

            2

        )

        self.window_size.setValue(

            ensure_odd(self.config.window_size)

        )

        self.k_value = QDoubleSpinBox()

        self.k_value.setRange(

            -1.0,

            1.0

        )

        self.k_value.setSingleStep(

            0.05

        )

        self.k_value.setValue(

            self.config.k

        )

        self.r_value = QSpinBox()

        self.r_value.setRange(

            1,

            255

        )

        self.r_value.setValue(

            self.config.r

        )

    # ------------------------------------------------------------------

    def create_layout(self):

        layout = QFormLayout()

        layout.addRow(

            "Method",

            self.method

        )

        layout.addRow(

            "Detect Dark Objects",

            self.dark_objects

        )

        layout.addRow(

            "Fixed Threshold",

            self.threshold

        )

        layout.addRow(

            "Value",

            self.value

        )

        layout.addRow(

            "Adaptive / Local Block Size",

            self.block_size

        )

        layout.addRow(

            "Constant C",

            self.constant_c

        )

        layout.addRow(

            "Window Size",

            self.window_size

        )

        layout.addRow(

            "k Value",

            self.k_value

        )

        layout.addRow(

            "Dynamic Range (r)",

            self.r_value

        )

        self.setLayout(layout)

    # ------------------------------------------------------------------

    def connect_signals(self):

        self.method.currentTextChanged.connect(

            self.method_changed

        )

        self.threshold.valueChanged.connect(

            self.slider_changed

        )

        self.dark_objects.toggled.connect(

            self.changed

        )

        self.block_size.valueChanged.connect(

            self.changed

        )

        self.constant_c.valueChanged.connect(

            self.changed

        )

        self.window_size.valueChanged.connect(

            self.changed

        )

        self.k_value.valueChanged.connect(

            self.changed

        )

        self.r_value.valueChanged.connect(

            self.changed

        )

    # ------------------------------------------------------------------

    def update_ui(self):

        method = self.method.currentText()

        is_fixed = method == "Fixed"

        is_adaptive = method in [

            "Adaptive Mean",

            "Adaptive Gaussian"

        ]

        is_local = method in [

            "Sauvola",

            "Niblack"

        ]

        self.threshold.setEnabled(is_fixed)

        self.value.setEnabled(is_fixed)

        self.block_size.setEnabled(is_adaptive or is_local)

        self.constant_c.setEnabled(is_adaptive or is_local)

        self.window_size.setEnabled(is_local)

        self.k_value.setEnabled(is_local)

        self.r_value.setEnabled(method == "Sauvola")

    # ------------------------------------------------------------------

    def slider_changed(self, value):

        self.value.setText(

            str(value)

        )

        self.changed()

    # ------------------------------------------------------------------

    def method_changed(self, value):

        self.config.threshold_method = value

        self.update_ui()

        self.parametersChanged.emit()

    # ------------------------------------------------------------------

    def changed(self):

        self.config.threshold_method = self.method.currentText()

        self.config.fixed_threshold = self.threshold.value()

        self.config.detect_dark_objects = self.dark_objects.isChecked()

        self.config.block_size = ensure_odd(

            self.block_size.value()

        )

        self.config.constant_c = self.constant_c.value()

        self.config.window_size = ensure_odd(

            self.window_size.value()

        )

        self.config.k = self.k_value.value()

        self.config.r = self.r_value.value()

        self.parametersChanged.emit()
###############################################################################
# Morphology Panel
###############################################################################


class MorphologyPanel(QGroupBox):

    parametersChanged = Signal()

    def __init__(self, config):

        super().__init__("Morphology")

        self.config = config

        self.create_widgets()

        self.create_layout()

        self.connect_signals()

    ###################################################################

    def create_widgets(self):

        self.chk_open = QCheckBox("Opening")
        self.chk_close = QCheckBox("Closing")
        self.chk_erode = QCheckBox("Erosion")
        self.chk_dilate = QCheckBox("Dilation")

        self.chk_open.setChecked(self.config.opening)
        self.chk_close.setChecked(self.config.closing)
        self.chk_erode.setChecked(self.config.erosion)
        self.chk_dilate.setChecked(self.config.dilation)

        self.kernel_shape = QComboBox()

        self.kernel_shape.addItems([

            "Ellipse",

            "Rectangle",

            "Cross"

        ])

        self.kernel_shape.setCurrentText(

            self.config.kernel_shape

        )

        self.kernel_size = QSpinBox()

        self.kernel_size.setRange(

            1,

            51

        )

        self.kernel_size.setSingleStep(

            2

        )

        self.kernel_size.setValue(

            self.config.kernel_size

        )

        self.iterations = QSpinBox()

        self.iterations.setRange(

            1,

            20

        )

        self.iterations.setValue(

            self.config.iterations

        )

    ###################################################################

    def create_layout(self):

        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        tab_order = [
            ("Opening", self.chk_open),
            ("Closing", self.chk_close),
            ("Erosion", self.chk_erode),
            ("Dilation", self.chk_dilate),
        ]

        for name, checkbox in tab_order:

            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.addWidget(checkbox)
            tab_layout.addStretch()
            self.tabs.addTab(tab, name)

        layout.addWidget(self.tabs)

        common_layout = QFormLayout()

        common_layout.addRow(

            "Kernel Shape",

            self.kernel_shape

        )

        common_layout.addRow(

            "Kernel Size",

            self.kernel_size

        )

        common_layout.addRow(

            "Iterations",

            self.iterations

        )

        layout.addLayout(common_layout)

        self.setLayout(layout)

    ###################################################################

    def connect_signals(self):

        self.chk_open.stateChanged.connect(

            self.changed

        )

        self.chk_close.stateChanged.connect(

            self.changed

        )

        self.chk_erode.stateChanged.connect(

            self.changed

        )

        self.chk_dilate.stateChanged.connect(

            self.changed

        )

        self.kernel_shape.currentTextChanged.connect(

            self.changed

        )

        self.kernel_size.valueChanged.connect(

            self.changed

        )

        self.iterations.valueChanged.connect(

            self.changed

        )

    ###################################################################

    def changed(self):

        self.config.opening = self.chk_open.isChecked()

        self.config.closing = self.chk_close.isChecked()

        self.config.erosion = self.chk_erode.isChecked()

        self.config.dilation = self.chk_dilate.isChecked()

        self.config.kernel_shape = self.kernel_shape.currentText()

        self.config.kernel_size = self.kernel_size.value()

        self.config.iterations = self.iterations.value()

        self.parametersChanged.emit()
###############################################################################
# Pandas Table Model
###############################################################################

class DataFrameModel(QAbstractTableModel):

    def __init__(self, dataframe=pd.DataFrame()):
        super().__init__()
        self._df = dataframe

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._df = dataframe
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):

        if not index.isValid():
            return None

        if role == Qt.DisplayRole:

            value = self._df.iloc[
                index.row(),
                index.column()
            ]

            if isinstance(value, float):
                return f"{value:.3f}"

            return str(value)

        return None

    def headerData(self, section, orientation, role):

        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self._df.columns[section]

        return str(section + 1)
###############################################################################
# Main Window
###############################################################################


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        # -------------------------------------------------------------
        # Core Objects
        # -------------------------------------------------------------

        self.config = AppConfig()

        self.pipeline = ProcessingPipeline()

        self.processor = ImageProcessor(

            self.pipeline,

            self.config

        )

        # -------------------------------------------------------------
        # Window
        # -------------------------------------------------------------

        self.setWindowTitle(

            f"{APP_NAME}  v{VERSION}"

        )

        self.resize(

            self.config.window_width,

            self.config.window_height

        )

        # -------------------------------------------------------------
        # Build
        # -------------------------------------------------------------

        self.create_actions()

        self.create_menu()

        self.create_widgets()

        self.create_layout()

        self.connect_signals()

        self.create_statusbar()

        self.viewer.show_placeholder()

    ###################################################################
    # Actions
    ###################################################################

    def create_actions(self):

        self.action_open = QAction(

            "Open Image...",

            self

        )

        self.action_open.setShortcut(

            "Ctrl+O"

        )

        self.action_save_image = QAction(

            "Save Current View...",

            self

        )

        self.action_export_measurements = QAction(

            "Export Measurements...",

            self

        )

        self.action_exit = QAction(

            "Exit",

            self

        )

        self.action_exit.setShortcut(

            "Ctrl+Q"

        )

        self.action_about = QAction(

            "About",

            self

        )

        self.action_fit = QAction(

            "Fit to Window",

            self

        )

        self.action_actual = QAction(

            "Actual Size",

            self

        )

    ###################################################################
    # Menu
    ###################################################################

    def create_menu(self):

        menu = self.menuBar()

        file_menu = menu.addMenu("File")

        file_menu.addAction(

            self.action_open

        )

        file_menu.addAction(

            self.action_save_image

        )

        file_menu.addSeparator()

        file_menu.addAction(

            self.action_exit

        )

        view_menu = menu.addMenu("View")

        view_menu.addAction(

            self.action_fit

        )

        view_menu.addAction(

            self.action_actual

        )

        export_menu = menu.addMenu("Export")

        export_menu.addAction(

            self.action_export_measurements

        )

        help_menu = menu.addMenu("Help")

        help_menu.addAction(

            self.action_about

        )

    ###################################################################
    # Widgets
    ###################################################################

    def create_widgets(self):

        self.central = QWidget()

        self.setCentralWidget(

            self.central

        )

        # -------------------------------------------------------------
        # Viewer
        # -------------------------------------------------------------

        self.viewer = ImageViewer()
        # -------------------------------------------------------------
        # Measurement Table
        # -------------------------------------------------------------

        self.measurement_table = QTableView()

        self.measurement_model = DataFrameModel()

        self.measurement_table.setModel(
            self.measurement_model
        )

        self.measurement_table.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )

        self.measurement_table.setAlternatingRowColors(True)

        self.measurement_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        self.measurement_table.verticalHeader().setVisible(False)

        # -------------------------------------------------------------
        # Pipeline
        # -------------------------------------------------------------

        self.pipeline_list = QListWidget()

        self.pipeline_list.addItems([

            "Original",

            "Gray",

            "CLAHE",

            "Binary",

            "Morphology",

            "Labels",

            "Overlay",

            "Measurements",

            "Statistics"

        ])

        # -------------------------------------------------------------
        # Groups
        # -------------------------------------------------------------

        self.pipeline_group = QGroupBox(

            "Pipeline"

        )

        p_layout = QVBoxLayout()

        p_layout.addWidget(

            self.pipeline_list

        )

        self.pipeline_group.setLayout(

            p_layout

        )

        self.viewer_group = QGroupBox(

            "Image Viewer"

        )

        v_layout = QVBoxLayout()

        v_layout.addWidget(

            self.viewer

        )

        self.viewer_group.setLayout(

            v_layout
        )

        # -------------------------------------------------------------
        # Parameters
        # -------------------------------------------------------------

        self.parameter_group = QGroupBox(

            "Parameters"

        )

        self.parameter_layout = QVBoxLayout()

        self.parameter_group.setLayout(

            self.parameter_layout

        )

        self.parameter_label = QLabel(

            "Select a pipeline stage."

        )

        self.parameter_layout.addWidget(

            self.parameter_label

        )

        self.parameter_layout.addStretch()

        # -------------------------------------------------------------
        # Results
        # -------------------------------------------------------------

        self.results_group = QGroupBox(

            "Results"

        )

        r_layout = QGridLayout()

        self.results_group.setLayout(

            r_layout

        )

        self.lbl_pores = QLabel("-")

        self.lbl_porosity = QLabel("-")

        self.lbl_area = QLabel("-")

        self.lbl_diameter = QLabel("-")

        r_layout.addWidget(

            QLabel("Pores"),

            0,

            0

        )

        r_layout.addWidget(

            self.lbl_pores,

            0,

            1

        )

        r_layout.addWidget(

            QLabel("Porosity"),

            1,

            0

        )

        r_layout.addWidget(

            self.lbl_porosity,

            1,

            1

        )

        r_layout.addWidget(

            QLabel("Mean Area"),

            2,

            0

        )

        r_layout.addWidget(

            self.lbl_area,

            2,

            1

        )

        r_layout.addWidget(

            QLabel("Mean Diameter"),

            3,

            0

        )

        r_layout.addWidget(

            self.lbl_diameter,

            3,

            1

        )
        ###################################################################
    # Layout
    ###################################################################

    def create_layout(self):

        self.main_layout = QVBoxLayout(self.central)

        # -------------------------------------------------------------
        # Top Splitter
        # -------------------------------------------------------------

        self.top_splitter = QSplitter(Qt.Horizontal)

        self.top_splitter.addWidget(
            self.pipeline_group
        )

        self.top_splitter.addWidget(
            self.viewer_group
        )

        self.top_splitter.addWidget(
            self.parameter_group
        )

        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 5)
        self.top_splitter.setStretchFactor(2, 2)

        self.main_layout.addWidget(
            self.top_splitter,
            stretch=8
        )

        # -------------------------------------------------------------
        # Bottom
        # -------------------------------------------------------------

        self.main_layout.addWidget(
            self.results_group,
            stretch=1
        )

    ###################################################################
    # Status Bar
    ###################################################################

    def create_statusbar(self):

        status = QStatusBar()

        self.setStatusBar(status)

        self.statusBar().showMessage("Ready")

    ###################################################################
    # Connections
    ###################################################################

    def connect_signals(self):

        self.action_open.triggered.connect(
            self.open_image
        )

        self.action_exit.triggered.connect(
            self.close
        )

        self.action_fit.triggered.connect(
            self.viewer.fit_to_window
        )

        self.action_actual.triggered.connect(
            self.viewer.actual_size
        )

        self.action_about.triggered.connect(
            self.about
        )

        self.pipeline_list.currentTextChanged.connect(
            self.display_stage
        )

    ###################################################################
    # Parameter Panel
    ###################################################################

    def clear_parameter_panel(self):

        while self.parameter_layout.count():

            item = self.parameter_layout.takeAt(0)

            widget = item.widget()

            if widget is not None:

                widget.deleteLater()

    # ------------------------------------------------------------------

    def show_default_panel(self):

        self.clear_parameter_panel()

        label = QLabel(
            "No parameters available."
        )

        label.setAlignment(Qt.AlignTop)

        self.parameter_layout.addWidget(label)

        self.parameter_layout.addStretch()

    # ------------------------------------------------------------------

    def show_clahe_panel(self):

        self.clear_parameter_panel()

        self.clahe_panel = CLAHEPanel(
            self.config
        )

        self.parameter_layout.addWidget(
            self.clahe_panel
        )

        self.parameter_layout.addStretch()

        self.clahe_panel.parametersChanged.connect(
            self.update_clahe
        )

    # ------------------------------------------------------------------

    def show_threshold_panel(self):

        self.clear_parameter_panel()

        self.threshold_panel = ThresholdPanel(
            self.config
        )

        self.parameter_layout.addWidget(
            self.threshold_panel
        )

        self.parameter_layout.addStretch()

        self.threshold_panel.parametersChanged.connect(
            self.update_threshold
        )
    ###################################################################
    # Morphology Panel
    ###################################################################

    def show_morphology_panel(self):

        self.clear_parameter_panel()

        self.morphology_panel = MorphologyPanel(

            self.config

        )

        self.parameter_layout.addWidget(

            self.morphology_panel

        )

        self.parameter_layout.addStretch()

        self.morphology_panel.parametersChanged.connect(

            self.update_morphology

        )
        

    ###################################################################
    # Open Image
    ###################################################################

    def open_image(self):

        filename, _ = QFileDialog.getOpenFileName(

            self,

            "Open Image",

            "",

            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"

        )

        if not filename:

            return

        try:

            image = self.processor.load_image(
                filename
            )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )

            return

        self.viewer.set_image(image)

        self.pipeline_list.setCurrentRow(0)

        self.statusBar().showMessage(
            Path(filename).name
        )
        ###################################################################
    # Pipeline Display
    ###################################################################

    def display_stage(self, stage):

        if not stage:
            return

        stage = stage.lower()

        # ----------------------------------------------------------
        # Original
        # ----------------------------------------------------------

        if stage == "original":

            self.show_default_panel()

            image = self.processor.original()

        # ----------------------------------------------------------
        # Gray
        # ----------------------------------------------------------

        elif stage == "gray":

            self.show_default_panel()

            image = self.processor.gray()

        # ----------------------------------------------------------
        # CLAHE
        # ----------------------------------------------------------

        elif stage == "clahe":

            self.show_clahe_panel()

            image = self.processor.clahe()

        # ----------------------------------------------------------
        # Binary
        # ----------------------------------------------------------

        elif stage == "binary":

            self.show_threshold_panel()

            image = self.processor.binary()

        # ----------------------------------------------------------
        # Morphology
        # ----------------------------------------------------------

        elif stage == "morphology":

            self.show_morphology_panel()

            image = self.processor.morphology()

        # ----------------------------------------------------------
        # Labels
        # ----------------------------------------------------------

        elif stage == "labels":

            self.show_default_panel()

            image = self.processor.labels()

        # ----------------------------------------------------------
        # Overlay
        # ----------------------------------------------------------

        elif stage == "overlay":

            self.show_default_panel()

            image = self.processor.overlay()

        # ----------------------------------------------------------
        # Measurements
        # ----------------------------------------------------------

        elif stage == "measurements":

            self.show_measurements()

            return

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        elif stage == "statistics":

            self.show_default_panel()

            self.refresh_results()

            return

        else:

            return

        if image is not None:
            if self.measurement_table.parent() is not None:

                self.viewer_group.layout().removeWidget(
                    self.measurement_table
                )

                self.measurement_table.hide()

                self.viewer_group.layout().addWidget(
                    self.viewer
                )

                self.viewer.show()
            self.viewer.set_image(image)

    ###################################################################
    # CLAHE Update
    ###################################################################

    def update_clahe(self):

        self.pipeline.invalidate_after("gray")

        image = self.processor.clahe()

        self.viewer.set_image(image)

        self.statusBar().showMessage(

            "CLAHE Updated"

        )

    ###################################################################
    # Threshold Update
    ###################################################################

    def update_threshold(self):

        self.pipeline.invalidate_after("clahe")

        image = self.processor.binary()

        self.viewer.set_image(image)

        self.statusBar().showMessage(

            "Threshold Updated"

        )
    ###################################################################
    # Morphology Update
    ###################################################################

    def update_morphology(self):

        self.pipeline.invalidate_after(

            "binary"

        )

        image = self.processor.morphology()

        self.viewer.set_image(

            image

        )

        self.statusBar().showMessage(

            "Morphology Updated"

        )
    ###################################################################
    # Measurements
    ###################################################################

    def show_measurements(self):

        df = self.processor.measurements()

        self.measurement_model.setDataFrame(df)

        self.viewer_group.layout().removeWidget(
            self.viewer
        )

        self.viewer.hide()

        self.viewer_group.layout().addWidget(
            self.measurement_table
        )

        self.measurement_table.show()

        self.refresh_results()

    ###################################################################
    # Results
    ###################################################################

    def refresh_results(self):

        stats = self.processor.statistics()

        if not stats:

            self.lbl_pores.setText("-")

            self.lbl_porosity.setText("-")

            self.lbl_area.setText("-")

            self.lbl_diameter.setText("-")

            return

        self.lbl_pores.setText(

            str(stats["Number of Pores"])

        )

        self.lbl_porosity.setText(

            f'{stats["Porosity (%)"]:.2f} %'

        )

        self.lbl_area.setText(

            f'{stats["Mean Area"]:.2f}'

        )

        self.lbl_diameter.setText(

            f'{stats["Mean Diameter"]:.2f}'

        )

    ###################################################################
    # Export Helpers
    ###################################################################

    def save_current_image(self):

        image = self.viewer.current()

        if image is None:

            QMessageBox.warning(

                self,

                "Save Image",

                "No image is available to save."

            )

            return

        filename, _ = QFileDialog.getSaveFileName(

            self,

            "Save Image",

            "",

            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;BMP Files (*.bmp)"

        )

        if not filename:

            return

        try:

            cv2.imwrite(

                filename,

                image

            )

            self.statusBar().showMessage(

                f"Image saved: {Path(filename).name}"

            )

        except Exception as e:

            QMessageBox.critical(

                self,

                "Save Image",

                str(e)

            )

    # ------------------------------------------------------------------

    def export_measurements(self):

        df = self.processor.measurements()

        if df.empty:

            QMessageBox.warning(

                self,

                "Export Measurements",

                "No measurement data is available to export."

            )

            return

        filename, _ = QFileDialog.getSaveFileName(

            self,

            "Export Measurements",

            "",

            "CSV Files (*.csv);;Excel Files (*.xlsx)"

        )

        if not filename:

            return

        try:

            if filename.lower().endswith(".xlsx"):

                df.to_excel(filename, index=False)

            else:

                df.to_csv(filename, index=False)

            self.statusBar().showMessage(

                f"Measurements exported: {Path(filename).name}"

            )

        except Exception as e:

            QMessageBox.critical(

                self,

                "Export Measurements",

                str(e)

            )

    ###################################################################
    # About
    ###################################################################

    def about(self):

        QMessageBox.about(

            self,

            APP_NAME,

            f"""

<h2>{APP_NAME}</h2>

Version {VERSION}

Professional Porosity Analysis Software

Built using

• PySide6

• OpenCV

• NumPy

• Pandas

• scikit-image

• Matplotlib

            """

        )
def main():

    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = MainWindow()

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":

    main()



