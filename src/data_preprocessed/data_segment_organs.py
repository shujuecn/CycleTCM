import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import PROCESSED_DATA_DIR

# Extract bounding box and width/height of foreground contour
def get_bounding_box_and_xy_delta(img_gray):
    _, binary = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Do not find foreground contours")
    largest = max(contours, key=cv2.contourArea)
    x_min, y_min, w, h = cv2.boundingRect(largest)

    return w, h, largest,x_min,y_min

def contour_points(largest):
    return largest[:, 0, :].astype(np.float32)  

def mean_x_of_topk_by_y(xs, ys, mask, k, pick='largest'):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    order = np.argsort(ys[idx])   
    if pick == 'largest':         
        order = order[::-1]
    take = idx[order[:min(k, order.size)]]
    return float(np.mean(xs[take]))

def mean_y_of_topk_by_x(xs, ys, mask, k, pick='largest'):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    order = np.argsort(xs[idx])   
    if pick == 'largest':         
        order = order[::-1]
    take = idx[order[:min(k, order.size)]]
    return float(np.mean(ys[take]))

def vertical_intersections_detailed(pts, x0, atol_colinear=True):
    res = []
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (x1 <= x0 < x2) or (x2 <= x0 < x1):
            t = (x0 - x1) / (x2 - x1 + 1e-12)
            y = y1 + t * (y2 - y1)
            res.append((float(y), i, float(t)))
        elif atol_colinear and (abs(x1 - x0) < 1e-6 and abs(x2 - x0) < 1e-6):
            res.append((float(y1), i, 0.0))
            res.append((float(y2), i, 1.0))
    return res

def horizontal_intersections_detailed(pts, y0, atol_colinear=True):
    res = []
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 <= y0 < y2) or (y2 <= y0 < y1):
            t = (y0 - y1) / (y2 - y1 + 1e-12)
            x = x1 + t * (x2 - x1)
            res.append((float(x), i, float(t)))
        elif atol_colinear and (abs(y1 - y0) < 1e-6 and abs(y2 - y0) < 1e-6):
            res.append((float(x1), i, 0.0))
            res.append((float(x2), i, 1.0))
    return res

def arc_between_prefer(pts, i1, t1, i2, t2, pref, x_c, y_c):
    N = len(pts)

    def P(i, t):
        p1, p2 = pts[i], pts[(i + 1) % N]
        return (1 - t) * p1 + t * p2

    p1 = P(i1, t1)
    p2 = P(i2, t2)

    fwd_idx = []
    j = (i1 + 1) % N
    while True:
        fwd_idx.append(j)
        if j == i2:
            break
        j = (j + 1) % N
    fwd_path = np.vstack([p1, pts[fwd_idx], p2])

    bwd_idx = []
    j = i1 % N
    while True:
        j = (j - 1 + N) % N
        bwd_idx.append(j)
        if j == (i2 + 1) % N:
            break
    bwd_path = np.vstack([p1, pts[bwd_idx], p2])

    if pref == 'upper':   
        score_f = lambda path: np.mean(path[:, 1] > y_c)
    elif pref == 'lower':
        score_f = lambda path: np.mean(path[:, 1] < y_c)
    elif pref == 'left':
        score_f = lambda path: np.mean(path[:, 0] < x_c)
    elif pref == 'right':
        score_f = lambda path: np.mean(path[:, 0] > x_c)
    else:
        score_f = lambda path: 0.0

    return fwd_path if score_f(fwd_path) >= score_f(bwd_path) else bwd_path

# Compute top edge arc
def compute_top_arc(pts, image_shape, k=160):
    H, W = image_shape[:2]
    x_c, y_c = W / 2.0, H / 2.0
    xs, ys = pts[:, 0], pts[:, 1]

    mask_up = ys > y_c  
    up1 = mean_x_of_topk_by_y(xs, ys, mask=(mask_up & (xs > x_c)), k=k, pick='largest')  
    up2 = mean_x_of_topk_by_y(xs, ys, mask=(mask_up & (xs < x_c)), k=k, pick='largest')  
    up1 = up1 if up1 is not None else x_c
    up2 = up2 if up2 is not None else x_c

    inter1 = vertical_intersections_detailed(pts, up1)
    inter2 = vertical_intersections_detailed(pts, up2)
    if len(inter1) == 0 or len(inter2) == 0:
        raise RuntimeError("top: vertical lines have no intersections")

    cand1 = [(y,i,t) for (y,i,t) in inter1 if y > y_c]
    cand2 = [(y,i,t) for (y,i,t) in inter2 if y > y_c]
    if len(cand1) == 0: cand1 = inter1
    if len(cand2) == 0: cand2 = inter2

    y1,i1,t1 = sorted(cand1, key=lambda z:z[0])[0]   
    y2,i2,t2 = sorted(cand2, key=lambda z:z[0])[0]

    arc = arc_between_prefer(pts, i1, t1, i2, t2, pref='upper', x_c=x_c, y_c=y_c)
    return up1, up2, arc

# Compute bottom edge arc
def compute_bottom_arc(pts, image_shape, k=160):
    H, W = image_shape[:2]
    x_c, y_c = W / 2.0, H / 2.0
    xs, ys = pts[:, 0], pts[:, 1]

    mask_dn = ys < y_c
    dn1 = mean_x_of_topk_by_y(xs, ys, mask=(mask_dn & (xs > x_c)), k=k, pick='smallest')  
    dn2 = mean_x_of_topk_by_y(xs, ys, mask=(mask_dn & (xs < x_c)), k=k, pick='smallest')  
    dn1 = dn1 if dn1 is not None else x_c
    dn2 = dn2 if dn2 is not None else x_c

    inter1 = vertical_intersections_detailed(pts, dn1)
    inter2 = vertical_intersections_detailed(pts, dn2)
    if len(inter1) == 0 or len(inter2) == 0:
        raise RuntimeError("bottom: vertical lines have no intersections")

    cand1 = [(y,i,t) for (y,i,t) in inter1 if y < y_c]
    cand2 = [(y,i,t) for (y,i,t) in inter2 if y < y_c]
    if len(cand1) == 0: cand1 = inter1
    if len(cand2) == 0: cand2 = inter2

    y1,i1,t1 = sorted(cand1, key=lambda z:z[0])[-1]
    y2,i2,t2 = sorted(cand2, key=lambda z:z[0])[-1]

    arc = arc_between_prefer(pts, i1, t1, i2, t2, pref='lower', x_c=x_c, y_c=y_c)
    return dn1, dn2, arc

# Compute left edge arc
def compute_left_arc(pts, image_shape, k=160):
    H, W = image_shape[:2]
    x_c, y_c = W / 2.0, H / 2.0
    xs, ys = pts[:, 0], pts[:, 1]

    y_top = mean_y_of_topk_by_x(xs, ys, mask=((xs < x_c) & (ys > y_c)), k=k, pick='smallest')  
    y_bot = mean_y_of_topk_by_x(xs, ys, mask=((xs < x_c) & (ys < y_c)), k=k, pick='smallest')  
    y_top = y_top if y_top is not None else y_c
    y_bot = y_bot if y_bot is not None else y_c

    inter1 = horizontal_intersections_detailed(pts, y_top)
    inter2 = horizontal_intersections_detailed(pts, y_bot)
    if len(inter1) == 0 or len(inter2) == 0:
        raise RuntimeError("left: horizontal lines have no intersections")

    cand1 = [(x,i,t) for (x,i,t) in inter1 if x < x_c]
    cand2 = [(x,i,t) for (x,i,t) in inter2 if x < x_c]
    if len(cand1) == 0: cand1 = inter1
    if len(cand2) == 0: cand2 = inter2

    x1,i1,t1 = sorted(cand1, key=lambda z:z[0])[-1]  
    x2,i2,t2 = sorted(cand2, key=lambda z:z[0])[-1]

    arc = arc_between_prefer(pts, i1, t1, i2, t2, pref='left', x_c=x_c, y_c=y_c)
    return y_top, y_bot, arc

# Compute right edge arc
def compute_right_arc(pts, image_shape, k=160):
    H, W = image_shape[:2]
    x_c, y_c = W / 2.0, H / 2.0
    xs, ys = pts[:, 0], pts[:, 1]

    y_top = mean_y_of_topk_by_x(xs, ys, mask=((xs > x_c) & (ys > y_c)), k=k, pick='largest')   
    y_bot = mean_y_of_topk_by_x(xs, ys, mask=((xs > x_c) & (ys < y_c)), k=k, pick='largest')   
    y_top = y_top if y_top is not None else y_c
    y_bot = y_bot if y_bot is not None else y_c

    inter1 = horizontal_intersections_detailed(pts, y_top)
    inter2 = horizontal_intersections_detailed(pts, y_bot)
    if len(inter1) == 0 or len(inter2) == 0:
        raise RuntimeError("right: horizontal lines have no intersections")

    cand1 = [(x,i,t) for (x,i,t) in inter1 if x > x_c]
    cand2 = [(x,i,t) for (x,i,t) in inter2 if x > x_c]
    if len(cand1) == 0: cand1 = inter1
    if len(cand2) == 0: cand2 = inter2

    x1,i1,t1 = sorted(cand1, key=lambda z:z[0])[0]  
    x2,i2,t2 = sorted(cand2, key=lambda z:z[0])[0]

    arc = arc_between_prefer(pts, i1, t1, i2, t2, pref='right', x_c=x_c, y_c=y_c)
    return y_top, y_bot, arc

# Apply curved erosion to arc
def erode_arc_curved(arc, ew, direction='inward', edge_type='top'):

    n = len(arc)
    if n == 0:
        return arc
    
    t = np.linspace(0, 1, n)  
    erosion_curve = 1 - (2 * t - 1) ** 2
    erosion_distances = erosion_curve * ew
    
    eroded_arc = arc.copy()
    
    if edge_type == 'top':
        eroded_arc[:, 1] = arc[:, 1] - erosion_distances
    elif edge_type == 'bottom':
        eroded_arc[:, 1] = arc[:, 1] + erosion_distances
    elif edge_type == 'left':
        eroded_arc[:, 0] = arc[:, 0] + erosion_distances
    elif edge_type == 'right':
        eroded_arc[:, 0] = arc[:, 0] - erosion_distances
    
    return eroded_arc, erosion_distances

def compute_center_rectangle_erosion(x_center, y_center, w, h, r2):

    erosion_x = int(w * r2 / 2)  
    erosion_y = int(h * r2 / 2)  
    
    x_left = x_center - erosion_x
    x_right = x_center + erosion_x
    y_top = y_center - erosion_y
    y_bottom = y_center + erosion_y
    
    rect_coords = np.array([
        [x_left, y_top],      
        [x_right, y_top],     
        [x_right, y_bottom],  
        [x_left, y_bottom]    
    ])
    
    erosion_distances = {
        'left': erosion_x,
        'right': erosion_x,
        'top': erosion_y,
        'bottom': erosion_y
    }
    
    return rect_coords, erosion_distances

# Create mask from closed region bounded by original arc and eroded arc
def create_mask_from_arc(arc_original, arc_eroded, img_shape):

    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    
    combined_points = np.vstack([arc_original, arc_eroded[::-1]])
    
    combined_points = combined_points.astype(np.int32)
    
    cv2.fillPoly(mask, [combined_points], 255)
    
    return mask


def create_ellipse_mask(w, h, top_sharpness=1.8, bottom_sharpness=1.3):

    y, x = np.ogrid[:h, :w]
    center_x, center_y = w / 2.0, h / 2.0
    a, b = w / 2.0, h / 2.0
    x_centered = x - center_x
    y_centered = y - center_y
    y_modified = np.where(
        y_centered < 0,
        y_centered * top_sharpness,
        y_centered * bottom_sharpness
    )
    ellipse_eq = (x_centered / a) ** 2 + (y_modified / b) ** 2
    mask = (ellipse_eq <= 1.0).astype(np.float64)
    return mask


def create_mask_from_rectangle(rect_coords, img_shape):

    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    
    rect_coords = rect_coords.astype(np.int32)
    
    cv2.fillPoly(mask, [rect_coords], 255)
    
    return mask

# Process single image and save heart/kidney/liver/spleen four regions
def process_single_image(image_path, output_dirs, r=0.168, r2=0.432, r_liver=None, k_top=160, k_bottom=160, k_left=200, k_right=200, spleen_top_sharpness=1.8, spleen_bottom_sharpness=1.3):

    try:
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if img_gray is None or img_color is None:
            print(f"  ✗ Do not read image: {image_path}")
            return False
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        w, h, largest, x_min, y_min = get_bounding_box_and_xy_delta(img_gray)
        contour_pts = contour_points(largest)
        x_max = x_min + w
        y_max = y_min + h
        
        M = cv2.moments(largest)
        if M['m00'] != 0:
            x_center = int(M['m10'] / M['m00'])
            y_center = int(M['m01'] / M['m00'])
        else:
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
        
        # Compute four edge arcs
        up1, up2, arc_top = compute_top_arc(contour_pts, img_gray.shape, k=k_top)
        dn1, dn2, arc_bottom = compute_bottom_arc(contour_pts, img_gray.shape, k=k_bottom)
        yL_top, yL_bot, arc_left = compute_left_arc(contour_pts, img_gray.shape, k=k_left)
        yR_top, yR_bot, arc_right = compute_right_arc(contour_pts, img_gray.shape, k=k_right)
        
        # Compute erosion distance
        ew = int(np.sqrt(h**2 + w**2) * r)
        ew_right = int(np.sqrt(h**2 + w**2) * (r_liver if r_liver is not None else r))

        arc_top_eroded, _ = erode_arc_curved(arc_top, ew, edge_type='top')
        arc_bottom_eroded, _ = erode_arc_curved(arc_bottom, ew, edge_type='bottom')
        arc_left_eroded, _ = erode_arc_curved(arc_left, ew, edge_type='left')
        arc_right_eroded, _ = erode_arc_curved(arc_right, ew_right, edge_type='right')
        
        center_rect, _ = compute_center_rectangle_erosion(x_center, y_center, w, h, r2)
        x_left = int(center_rect[:, 0].min())
        x_right = int(center_rect[:, 0].max())
        y_top = int(center_rect[:, 1].min())
        y_bottom = int(center_rect[:, 1].max())
        w_rect = x_right - x_left
        h_rect = y_bottom - y_top
        if w_rect > 0 and h_rect > 0:
            ellipse_in_rect = create_ellipse_mask(w_rect, h_rect, spleen_top_sharpness, spleen_bottom_sharpness)
            H, W = img_gray.shape[:2]
            mask_center = np.zeros((H, W), dtype=np.uint8)
            mask_center[y_top:y_bottom, x_left:x_right] = (ellipse_in_rect * 255).astype(np.uint8)
        else:
            mask_center = create_mask_from_rectangle(center_rect, img_gray.shape)
        
        # Create masks for each region
        mask_top = create_mask_from_arc(arc_top, arc_top_eroded, img_gray.shape)
        mask_bottom = create_mask_from_arc(arc_bottom, arc_bottom_eroded, img_gray.shape)
        mask_right = create_mask_from_arc(arc_right, arc_right_eroded, img_gray.shape)
        
        # Extract each region from original image
        region_top = cv2.bitwise_and(img_color, img_color, mask=mask_top)
        region_bottom = cv2.bitwise_and(img_color, img_color, mask=mask_bottom)
        region_right = cv2.bitwise_and(img_color, img_color, mask=mask_right)
        region_center = cv2.bitwise_and(img_color, img_color, mask=mask_center)
        
        # Save each region
        cv2.imwrite(os.path.join(output_dirs['top_edge'], f"{base_filename}_heart_lung.png"), region_top)
        cv2.imwrite(os.path.join(output_dirs['bottom_edge'], f"{base_filename}_kidney.png"), region_bottom)
        cv2.imwrite(os.path.join(output_dirs['right_edge'], f"{base_filename}_liver.png"), region_right)
        cv2.imwrite(os.path.join(output_dirs['center_rect'], f"{base_filename}_spleen.png"), region_center)
        
        print(f"  ✓ finished: {base_filename}")
        return True
        
    except Exception as e:
        print(f"  ✗ failed {os.path.basename(image_path)}: {e}")
        return False


def batch_process_images(input_dir, output_base_dir=str(PROCESSED_DATA_DIR), r=0.168, r2=0.432, r_liver=None, spleen_top_sharpness=1.8, spleen_bottom_sharpness=1.3):

    output_dirs = {
        'top_edge': os.path.join(output_base_dir, 'images_heart_lung'),
        'bottom_edge': os.path.join(output_base_dir, 'images_kidney'),
        'right_edge': os.path.join(output_base_dir, 'images_liver'),
        'center_rect': os.path.join(output_base_dir, 'images_spleen')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    
    if not image_files:
        print(f"✗ Do not find PNG images in {input_dir}")
        return
    
    success_count = 0
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {image_file}")
        image_path = os.path.join(input_dir, image_file)
        
        if process_single_image(image_path, output_dirs, r=r, r2=r2, r_liver=r_liver, spleen_top_sharpness=spleen_top_sharpness, spleen_bottom_sharpness=spleen_bottom_sharpness):
            success_count += 1

    print(f"Success: {success_count}/{len(image_files)}")
    print(f"Failure: {len(image_files) - success_count}/{len(image_files)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Organ-associated tongue region segmentation')
    parser.add_argument('--input-dir', default=str(PROCESSED_DATA_DIR / 'pp'))
    parser.add_argument('--output-dir', default=str(PROCESSED_DATA_DIR))
    parser.add_argument('--r', type=float, default=0.196, help='Edge erosion rate')
    parser.add_argument('--r2', type=float, default=0.632, help='Center rectangle erosion rate')
    parser.add_argument('--r-liver', type=float, default=0.10, help='Liver erosion rate')
    args = parser.parse_args()
    batch_process_images(args.input_dir, args.output_dir, r=args.r, r2=args.r2, r_liver=args.r_liver)
