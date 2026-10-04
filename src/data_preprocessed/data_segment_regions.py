import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import PROCESSED_DATA_DIR

# Get bounding box and width/height from binary contour, return largest contour
def get_bounding_box_and_xy_delta(img_gray):

    _, binary = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)  

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    largest = max(contours, key=cv2.contourArea)
    
    x_min, y_min, w, h = cv2.boundingRect(largest)
    
    x_max = x_min + w
    y_max = y_min + h

    x_delta = w
    y_delta = h
    
    return x_delta, y_delta, largest

# Segment tongue body/edge for single image and save body and edge images
def process_image(image_path, output_body_dir, output_edge_dir, r):
    if not os.path.exists(output_edge_dir):
        os.makedirs(output_edge_dir)
    if not os.path.exists(output_body_dir):
        os.makedirs(output_body_dir)

    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    x_delta, y_delta,largest = get_bounding_box_and_xy_delta(gray_image)

    mask = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, (255), thickness=cv2.FILLED)
    Mf = mask

    e_w = int(np.sqrt(x_delta**2 + y_delta**2) * r)

    kernel = np.ones((int(e_w), int(e_w)), np.uint8)

    Me = cv2.erode(Mf, kernel)

    Medge = cv2.subtract(Mf, Me)

    Mbody = cv2.subtract(Mf, Medge)

    Medge_region = cv2.bitwise_and(image, image, mask=Medge)
    Mbody_region = cv2.bitwise_and(image, image, mask=Mbody)

    base_filename = os.path.splitext(os.path.basename(image_path))[0]  
    
    body_output_path = os.path.join(output_body_dir, f"{base_filename}_body.png")
    cv2.imwrite(body_output_path, Mbody_region)

    edge_output_path = os.path.join(output_edge_dir, f"{base_filename}_edge.png")
    cv2.imwrite(edge_output_path, Medge_region)

    print(f"Saved body image to: {body_output_path}")
    print(f"Saved edge image to: {edge_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Tongue Edge Segmentation')
    parser.add_argument('--r', type=float, required=True, help='Tongue edge width ratio')
    parser.add_argument('--input_dir', type=str, default=str(PROCESSED_DATA_DIR / 'pp'), help='Directory containing input images')
    parser.add_argument('--output_body_dir', type=str, default=str(PROCESSED_DATA_DIR / 'images_body'), help='Directory to save body images')
    parser.add_argument('--output_edge_dir', type=str, default=str(PROCESSED_DATA_DIR / 'images_edge'), help='Directory to save edge images')

    args = parser.parse_args()

    os.makedirs(args.output_body_dir, exist_ok=True)
    os.makedirs(args.output_edge_dir, exist_ok=True)

    input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.png')]

    for input_file in input_files:
        image_path = os.path.join(args.input_dir, input_file)
        process_image(image_path, args.output_body_dir, args.output_edge_dir, args.r)

if __name__ == "__main__":
    main()
