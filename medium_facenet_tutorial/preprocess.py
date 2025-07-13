import argparse
import glob
import logging
import multiprocessing as mp
import os
import time

import cv2

from medium_facenet_tutorial.align_dlib import AlignDlib

logger = logging.getLogger(__name__)

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))


def main(input_dir, output_dir, crop_dim):
    start_time = time.time()
    pool = mp.Pool(processes=mp.cpu_count())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
    for index, image_path in enumerate(image_paths):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
        output_path = os.path.join(image_output_dir, os.path.basename(image_path))
        pool.apply_async(preprocess_image, (image_path, output_path, crop_dim))

    pool.close()
    pool.join()
    logger.info('Completed in {} seconds'.format(time.time() - start_time))


def preprocess_image(input_path, output_path, crop_dim):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    image = _process_image(input_path, crop_dim)
    if image is not None:
        logger.debug('Writing processed file: {}'.format(output_path))
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    else:
        logger.warning("Skipping filename: {}".format(input_path))


def _process_image(filename, crop_dim):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        aligned_image = _align_image(image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _align_image(image, crop_dim):
    # Obtenir la plus grande boîte englobante du visage
    bb = align_dlib.getLargestFaceBoundingBox(image)
    if bb is None:
        return None

    # Extraire coordonnées brutes de la bounding box
    x, y, w, h = bb.left(), bb.top(), bb.width(), bb.height()

    # ✅ Ajouter une marge (par exemple : 30% autour du visage)
    margin_ratio = 0.3
    x_margin = int(w * margin_ratio)
    y_margin = int(h * margin_ratio)

    # Calculer coordonnées du cadrage élargi
    x1 = max(x - x_margin, 0)
    y1 = max(y - int(y_margin * 1.5), 0)  # ⚠️ on ajoute + de marge au dessus pour voir le front
    x2 = min(x + w + x_margin, image.shape[1])
    y2 = min(y + h + y_margin, image.shape[0])

    # Découper le visage élargi
    cropped_face = image[y1:y2, x1:x2]

    # Redimensionner vers `crop_dim x crop_dim`
    aligned = cv2.resize(cropped_face, (crop_dim, crop_dim))

    return aligned




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--input-dir', type=str, action='store', default='data', dest='input_dir')
    parser.add_argument('--output-dir', type=str, action='store', default='output', dest='output_dir')
    parser.add_argument('--crop-dim', type=int, action='store', default=224, dest='crop_dim',
                        help='Size to crop images to')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.crop_dim)
