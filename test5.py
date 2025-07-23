import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
import argparse
from pathlib import Path
import math
from typing import List, Tuple, Dict, Optional
import time
import re

class CompleteImprovedGPSFlightPathStitcher:
    def __init__(self, folder_path: str, output_path: str = "complete_improved_flight"):
        self.folder_path = Path(folder_path)
        self.output_path = output_path
        self.image_data = []
        self.straight_sections = []
        self.turn_sections = []
        self.section_images = []
        self.turn_images = []
        
    def extract_gps_coords(self, image_path: str) -> Optional[Tuple[float, float, float]]:
        """Extract GPS coordinates from image EXIF data"""
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if not exif_data:
                    return None
                
                gps_info = {}
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == "GPSInfo":
                        for gps_tag, gps_value in value.items():
                            gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                            gps_info[gps_tag_name] = gps_value
                
                if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
                    lat = self._convert_gps_coord(gps_info['GPSLatitude'])
                    lon = self._convert_gps_coord(gps_info['GPSLongitude'])
                    alt = self._convert_gps_coord(gps_info.get('GPSAltitude', 0))
                    
                    if gps_info.get('GPSLatitudeRef') == 'S':
                        lat = -lat
                    if gps_info.get('GPSLongitudeRef') == 'W':
                        lon = -lon
                    
                    return (lat, lon, alt)
        except Exception:
            pass
        return None
    
    def _convert_gps_coord(self, coord_tuple):
        """Convert GPS coordinate handling IFDRational"""
        try:
            if hasattr(coord_tuple, '__len__') and len(coord_tuple) == 3:
                degrees, minutes, seconds = coord_tuple
                return float(degrees) + float(minutes)/60 + float(seconds)/3600
            elif hasattr(coord_tuple, '__len__') and len(coord_tuple) == 1:
                return float(coord_tuple[0])
            else:
                return float(coord_tuple)
        except:
            return 0
    
    def load_flight_data(self):
        """Load flight data from images"""
        print("Loading flight data from images...")
        
        jpg_files = list(self.folder_path.glob("*.jpg")) + list(self.folder_path.glob("*.JPG"))
        jpg_files = sorted(self.folder_path.glob("*.jpg"), key=lambda x: x.name)
        #jpg_files.sort(key=lambda x: x.name)
        
        for img_file in jpg_files:
            try:
                gps_coords = self.extract_gps_coords(str(img_file))
                
                if gps_coords:
                    self.image_data.append({
                        'filename': img_file.name,
                        'path': str(img_file),
                        'lat': gps_coords[0],
                        'lon': gps_coords[1],
                        'alt': gps_coords[2],
                        'sequence': len(self.image_data) + 1
                    })
                    
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
        
        print(f"Loaded {len(self.image_data)} images with GPS data")
        return len(self.image_data) > 0
    
    def detect_turns_and_straight_sections(self):
        """Detect turn images vs straight flight images"""
        if len(self.image_data) < 5:
            return False
        
        print("Analyzing flight pattern to detect turns and straight sections...")
        
        bearings = []
        distances = []
        
        for i in range(1, len(self.image_data)):
            prev_point = self.image_data[i-1]
            curr_point = self.image_data[i]
            
            lat1, lon1 = np.radians(prev_point['lat']), np.radians(prev_point['lon'])
            lat2, lon2 = np.radians(curr_point['lat']), np.radians(curr_point['lon'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371000 * c
            distances.append(distance)
            
            y = np.sin(dlon) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
            bearings.append(bearing)
        
        turn_indicators = []
        
        for i in range(len(bearings)):
            is_turn = False
            
            if i > 0:
                angle_change = abs(bearings[i] - bearings[i-1])
                if angle_change > 180:
                    angle_change = 360 - angle_change
                if angle_change > 20:
                    is_turn = True
            
            if distances[i] < 8 or distances[i] > 35:
                is_turn = True
            
            if i >= 2 and i < len(bearings) - 2:
                prev_avg = np.mean([bearings[i-2], bearings[i-1]])
                next_avg = np.mean([bearings[i+1], bearings[i+2]]) if i+2 < len(bearings) else bearings[i+1]
                
                curr_vs_prev = abs(bearings[i] - prev_avg)
                curr_vs_next = abs(bearings[i] - next_avg)
                
                if curr_vs_prev > 180:
                    curr_vs_prev = 360 - curr_vs_prev
                if curr_vs_next > 180:
                    curr_vs_next = 360 - curr_vs_next
                
                if curr_vs_prev > 25 or curr_vs_next > 25:
                    is_turn = True
            
            turn_indicators.append(is_turn)
        
        for i, img_data in enumerate(self.image_data):
            if i == 0:
                img_data['type'] = 'STRAIGHT'
            elif i <= len(turn_indicators):
                idx = i - 1
                img_type = 'TURN' if turn_indicators[idx] else 'STRAIGHT'
                img_data['type'] = img_type
            else:
                img_data['type'] = 'STRAIGHT'
        
        turn_count = len([img for img in self.image_data if img.get('type') == 'TURN'])
        straight_count = len([img for img in self.image_data if img.get('type') == 'STRAIGHT'])
        print(f"Classification: {straight_count} straight, {turn_count} turn images")
        
        return True
    
    def group_straight_sections(self):
        """Group consecutive straight images into flight lines"""
        print("\nGrouping straight images into sections...")
        
        straight_images = [img for img in self.image_data if img.get('type') == 'STRAIGHT']
        if len(straight_images) < 2:
            return False
        
        self.straight_sections = []
        current_section = [straight_images[0]]
        
        for i in range(1, len(straight_images)):
            prev_img = straight_images[i-1]
            curr_img = straight_images[i]
            sequence_gap = curr_img['sequence'] - prev_img['sequence']
            
            should_break = False
            if sequence_gap > 4:
                should_break = True
            elif len(current_section) >= 2:
                section_start = current_section[0]
                section_end = current_section[-1]
                
                lat1_sec, lon1_sec = np.radians(section_start['lat']), np.radians(section_start['lon'])
                lat2_sec, lon2_sec = np.radians(section_end['lat']), np.radians(section_end['lon'])
                
                dlon_sec = lon2_sec - lon1_sec
                y_sec = np.sin(dlon_sec) * np.cos(lat2_sec)
                x_sec = np.cos(lat1_sec) * np.sin(lat2_sec) - np.sin(lat1_sec) * np.cos(lat2_sec) * np.cos(dlon_sec)
                section_bearing = (np.degrees(np.arctan2(y_sec, x_sec)) + 360) % 360
                
                lat1, lon1 = np.radians(prev_img['lat']), np.radians(prev_img['lon'])
                lat2, lon2 = np.radians(curr_img['lat']), np.radians(curr_img['lon'])
                dlon = lon2 - lon1
                y = np.sin(dlon) * np.cos(lat2)
                x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
                bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
                
                bearing_diff = abs(section_bearing - bearing)
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff
                if bearing_diff > 30:
                    should_break = True
            
            if should_break:
                if len(current_section) >= 2:
                    self.straight_sections.append(current_section)
                current_section = [curr_img]
            else:
                current_section.append(curr_img)
        
        if len(current_section) >= 2:
            self.straight_sections.append(current_section)
        
        # Process sections
        for i, section in enumerate(self.straight_sections):
            section_ordered = self.order_images_by_gps_progression(section)
            
            first_img = section_ordered[0]
            last_img = section_ordered[-1]
            
            lat1, lon1 = np.radians(first_img['lat']), np.radians(first_img['lon'])
            lat2, lon2 = np.radians(last_img['lat']), np.radians(last_img['lon'])
            
            dlon = lon2 - lon1
            y = np.sin(dlon) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
            
            if 22.5 <= bearing < 67.5:
                direction = "NE"
            elif 67.5 <= bearing < 112.5:
                direction = "E"
            elif 112.5 <= bearing < 157.5:
                direction = "SE"
            elif 157.5 <= bearing < 202.5:
                direction = "S"
            elif 202.5 <= bearing < 247.5:
                direction = "SW"
            elif 247.5 <= bearing < 292.5:
                direction = "W"
            elif 292.5 <= bearing < 337.5:
                direction = "NW"
            else:
                direction = "N"
            
            section_info = {
                'section_id': i + 1,
                'images': section_ordered,
                'start_seq': section_ordered[0]['sequence'],
                'end_seq': section_ordered[-1]['sequence'],
                'direction': direction,
                'bearing': bearing,
                'image_count': len(section_ordered)
            }
            
            self.straight_sections[i] = section_info
        
        print(f"Created {len(self.straight_sections)} straight sections")
        return len(self.straight_sections) > 0
    
    def group_turn_sections(self):
        """Group consecutive turn images into turn sections"""
        print("Grouping turn images into sections...")
        
        turn_images = [img for img in self.image_data if img.get('type') == 'TURN']
        if len(turn_images) < 2:
            return False
        
        self.turn_sections = []
        current_section = [turn_images[0]]
        
        for i in range(1, len(turn_images)):
            prev_img = turn_images[i-1]
            curr_img = turn_images[i]
            sequence_gap = curr_img['sequence'] - prev_img['sequence']
            
            if sequence_gap <= 5:
                current_section.append(curr_img)
            else:
                if len(current_section) >= 2:
                    self.turn_sections.append(current_section)
                current_section = [curr_img]
        
        if len(current_section) >= 2:
            self.turn_sections.append(current_section)
        
        for i, section in enumerate(self.turn_sections):
            section_ordered = sorted(section, key=lambda x: x['sequence'])
            
            center_lat = np.mean([img['lat'] for img in section_ordered])
            center_lon = np.mean([img['lon'] for img in section_ordered])
            
            first_img = section_ordered[0]
            last_img = section_ordered[-1]
            
            lat1, lon1 = np.radians(first_img['lat']), np.radians(first_img['lon'])
            lat2, lon2 = np.radians(last_img['lat']), np.radians(last_img['lon'])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            displacement = np.sqrt(dlat**2 + dlon**2) * 6371000
            
            if displacement < 50:
                turn_type = "STATIONARY"
            elif displacement < 150:
                turn_type = "TIGHT_TURN"
            else:
                turn_type = "WIDE_TURN"
            
            section_info = {
                'section_id': i + 1,
                'images': section_ordered,
                'start_seq': section_ordered[0]['sequence'],
                'end_seq': section_ordered[-1]['sequence'],
                'turn_type': turn_type,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'displacement': displacement,
                'image_count': len(section_ordered)
            }
            
            self.turn_sections[i] = section_info
        
        print(f"Created {len(self.turn_sections)} turn sections")
        return len(self.turn_sections) > 0
    
    def order_images_by_gps_progression(self, images):
        """Order images by their GPS progression along the flight line"""
        if len(images) <= 2:
            return images
        
        first_img = images[0]
        last_img = images[-1]
        
        lat1, lon1 = np.radians(first_img['lat']), np.radians(first_img['lon'])
        lat2, lon2 = np.radians(last_img['lat']), np.radians(last_img['lon'])
        
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        overall_bearing = np.arctan2(y, x)
        
        projections = []
        
        for img in images:
            lat, lon = np.radians(img['lat']), np.radians(img['lon'])
            dlat = lat - lat1
            dlon = lon - lon1
            projection = dlat * np.cos(overall_bearing) + dlon * np.sin(overall_bearing)
            projections.append((projection, img))
        
        projections.sort(key=lambda x: x[0])
        return [img for _, img in projections]
    
    def load_single_image(self, img_data, target_size=800):
        """Load and resize a single image"""
        try:
            img = cv2.imread(img_data['path'])
            if img is None:
                return None
            
            h, w = img.shape[:2]
            ratio = min(target_size/w, target_size/h)
            
            if ratio < 1:
                new_w, new_h = int(w*ratio), int(h*ratio)
                
                # Use LANCZOS for better quality when downsampling
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Apply subtle sharpening to counter resize blur
                kernel = np.array([[-0.5, -0.5, -0.5],
                                   [-0.5,  5.0, -0.5],
                                   [-0.5, -0.5, -0.5]])
                img = cv2.filter2D(img, -1, kernel)
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            return img
            
        except Exception as e:
            print(f"Error loading {img_data['filename']}: {e}")
            return None
    
    def calculate_gps_distance(self, img1_data, img2_data):
        """Calculate distance between two images using GPS coordinates"""
        lat1, lon1 = np.radians(img1_data['lat']), np.radians(img1_data['lon'])
        lat2, lon2 = np.radians(img2_data['lat']), np.radians(img2_data['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371000 * c
    
    def validate_homography_with_gps(self, M, img1_data, img2_data, img1_shape, img2_shape):
        """More lenient GPS validation"""
        if M is None:
            return False, "No homography matrix"
        
        # Stricter homography validation to prevent artifacts
        if M is None:
            return False, "No homography matrix"
        
        # Check for degenerate transformations
        det = np.linalg.det(M[:2, :2])
        if abs(det) < 0.1 or abs(det) > 10:
            return False, f"Degenerate transformation: det={det:.3f}"
        
        # Check perspective distortion
        if abs(M[2, 0]) > 0.002 or abs(M[2, 1]) > 0.002:
            return False, "Excessive perspective distortion"
        
        # Validate corner transformations
        h2, w2 = img2_shape[:2]
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        
        try:
            transformed_corners = cv2.perspectiveTransform(corners, M)
        except:
            return False, "Perspective transform failed"
        
        # Check for reasonable corner positions
        for corner in transformed_corners:
            x, y = corner[0]
            if abs(x) > 2000 or abs(y) > 2000:  # Stricter bounds
                return False, "Extreme corner transformation"
        
        # Check area preservation
        original_area = w2 * h2
        transformed_area = cv2.contourArea(transformed_corners)
        
        if transformed_area <= 0:
            return False, "Negative/zero transformed area"
        
        area_ratio = transformed_area / original_area
        if area_ratio < 0.3 or area_ratio > 3.0:  # Stricter area bounds
            return False, f"Excessive area distortion: {area_ratio:.2f}"
        
        return True, "Valid transformation"
    
    def filter_matches_geometrically(self, matches, kp1, kp2, img1_data, img2_data):
        """Fixed geometric filtering that's less restrictive"""
        if len(matches) < 10:
            return matches
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Calculate GPS bearing (keep this for reference)
        lat1, lon1 = np.radians(img1_data['lat']), np.radians(img1_data['lon'])
        lat2, lon2 = np.radians(img2_data['lat']), np.radians(img2_data['lon'])
        
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        gps_bearing = np.degrees(np.arctan2(y, x))
        
        # More lenient filtering approach
        filtered_matches = []
        displacements = []
        
        # First pass: collect displacement statistics
        for i, match in enumerate(matches):
            dx = pts2[i][0] - pts1[i][0]
            dy = pts2[i][1] - pts1[i][1]
            displacement = np.sqrt(dx*dx + dy*dy)
            
            if displacement > 3:  # Reduced from 5
                displacements.append((displacement, dx, dy, match, i))
        
        if len(displacements) < 5:
            return matches  # Return all if too few to analyze
        
        # Calculate median displacement direction for consistency
        dx_values = [d[1] for d in displacements]
        dy_values = [d[2] for d in displacements]
        median_dx = np.median(dx_values)
        median_dy = np.median(dy_values)
        median_angle = np.degrees(np.arctan2(median_dx, -median_dy))
        
        # Second pass: filter based on consistency with median
        for displacement, dx, dy, match, idx in displacements:
            # Check consistency with median displacement direction
            angle = np.degrees(np.arctan2(dx, -dy))
            angle_diff = abs(angle - median_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # MUCH more lenient angle filtering
            if angle_diff < 150:  # Increased from 90 to 150
                filtered_matches.append(match)
                continue
            
            # Alternative: GPS bearing check (also more lenient)
            bearing_diff = abs(angle - gps_bearing)
            if bearing_diff > 180:
                bearing_diff = 360 - bearing_diff
            
            if bearing_diff < 120:  # Increased from 90 to 120
                filtered_matches.append(match)
        
        # Fallback: if we filtered out too many, use statistical outlier removal instead
        if len(filtered_matches) < max(10, len(matches) * 0.3):
            print(f"    GPS filtering too strict ({len(filtered_matches)}/{len(matches)}), using statistical filtering")
            
            # Statistical outlier removal based on displacement magnitude
            displacements_only = [d[0] for d in displacements]
            q75, q25 = np.percentile(displacements_only, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            filtered_matches = []
            for displacement, dx, dy, match, idx in displacements:
                if lower_bound <= displacement <= upper_bound:
                    filtered_matches.append(match)
        
        return filtered_matches
    
  # Add this enhanced debugging method to your class

    def find_optimal_seam(self, img1, img2, overlap_region):
        """Find optimal seam line in overlap region using dynamic programming"""
        h, w = overlap_region.shape[:2]
        
        # Convert to grayscale for seam finding
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference map
        diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
        
        # Apply Gaussian blur to smooth the difference
        diff = cv2.GaussianBlur(diff, (5, 5), 1.0)
        
        # Dynamic programming for vertical seam
        energy = diff.copy()
        dp = np.zeros_like(energy)
        dp[0] = energy[0]
        
        for i in range(1, h):
            for j in range(w):
                # Consider three possible paths
                candidates = [dp[i-1, j]]  # Straight down
                if j > 0:
                    candidates.append(dp[i-1, j-1])  # Diagonal left
                if j < w-1:
                    candidates.append(dp[i-1, j+1])  # Diagonal right
                
                dp[i, j] = energy[i, j] + min(candidates)
        
        # Backtrack to find seam
        seam = np.zeros(h, dtype=np.int32)
        seam[-1] = np.argmin(dp[-1])
        
        for i in range(h-2, -1, -1):
            j = seam[i+1]
            candidates = [j]
            if j > 0:
                candidates.append(j-1)
            if j < w-1:
                candidates.append(j+1)
            
            best_j = candidates[np.argmin([dp[i, c] for c in candidates])]
            seam[i] = best_j
        
        return seam

    def enhanced_blend_with_seam(self, img1, img2, img1_region, img2_region, overlap_mask):
        """Enhanced blending using seam finding and multi-band blending"""
        h, w = img1_region.shape[:2]
        
        if not np.any(overlap_mask):
            return img1_region
        
        # Find optimal seam
        seam = self.find_optimal_seam(img1_region, img2_region, overlap_mask)
        
        # Create smooth transition mask
        blend_mask = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            # Create smooth transition around seam
            seam_pos = seam[i]
            for j in range(w):
                if j < seam_pos - 10:
                    blend_mask[i, j] = 1.0  # Fully img1
                elif j > seam_pos + 10:
                    blend_mask[i, j] = 0.0  # Fully img2
                else:
                    # Smooth transition
                    dist_from_seam = abs(j - seam_pos)
                    blend_mask[i, j] = 1.0 - (dist_from_seam / 20.0)
                    blend_mask[i, j] = np.clip(blend_mask[i, j], 0.0, 1.0)
        
        # Apply Gaussian smoothing to blend mask
        blend_mask = cv2.GaussianBlur(blend_mask, (15, 15), 5.0)
        
        # Blend images
        result = np.zeros_like(img1_region)
        for c in range(3):
            result[:, :, c] = (
                blend_mask * img1_region[:, :, c] + 
                (1 - blend_mask) * img2_region[:, :, c]
            ).astype(np.uint8)
        
        # Return blended region
        return result

    def improved_feature_matching(self, img1, img2, timeout_seconds=20):
        """Improved SIFT matching with better filtering"""
        start_time = time.time()
        
        # Create SIFT with optimized parameters for aerial imagery
        sift = cv2.SIFT_create(
            nfeatures=8000,           # More features for better coverage
            contrastThreshold=0.02,   # Lower threshold for low-contrast areas
            edgeThreshold=6,          # Reduced edge threshold
            sigma=1.2                 # Slightly reduced sigma
        )
        
        # Detect keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 50 or len(des2) < 50:
            return None, None, []
        
        # Use FLANN matcher with optimized parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=12)
        search_params = dict(checks=200)  # Increased checks for accuracy
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test with stricter ratio
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.6 * n.distance:  # Stricter ratio
                    good_matches.append(m)
        
        return kp1, kp2, good_matches

    def stitch_two_images_improved_debug(self, img1, img2, img1_data, img2_data, timeout_seconds=20):
        """Debug version of image stitching with detailed failure reporting"""
        start_time = time.time()
        section_id = img1_data.get('section_id', 'unknown')
        
        print(f"    DEBUG: Stitching images {img1_data['filename']} + {img2_data['filename']}")
        
        try:
            kp1, kp2, good_matches = self.improved_feature_matching(img1, img2, timeout_seconds)
            
            if kp1 is None or kp2 is None or len(good_matches) < 15:
                print(f"    ✗ INSUFFICIENT GOOD MATCHES: {len(good_matches) if good_matches else 0} < 15")
                return None
            
            print(f"    Features: img1={len(kp1)}, img2={len(kp2)}")
            print(f"    Good matches: {len(good_matches)}")
            
            filtered_matches = self.filter_matches_geometrically(good_matches, kp1, kp2, img1_data, img2_data)
            print(f"    Filtered matches: {len(filtered_matches)}")
            
            if len(filtered_matches) < 10:
                print(f"    ✗ INSUFFICIENT FILTERED MATCHES: {len(filtered_matches)} < 10")
                return None
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            
            best_M = None
            best_inliers = 0
            
            for attempt in range(3):
                M, mask = cv2.findHomography(
                    dst_pts, src_pts, 
                    cv2.RANSAC, 
                    ransacReprojThreshold=2.0 + attempt * 0.5,
                    maxIters=5000,
                    confidence=0.995
                )
                
                if M is not None:
                    inliers = np.sum(mask)
                    print(f"    Homography attempt {attempt+1}: {inliers} inliers")
                    if inliers > best_inliers:
                        best_M = M
                        best_inliers = inliers
            
            M = best_M
            
            if M is None:
                print(f"    ✗ NO VALID HOMOGRAPHY found")
                return None
            
            print(f"    Best homography: {best_inliers} inliers")
            
            is_valid, reason = self.validate_homography_with_gps(M, img1_data, img2_data, img1.shape, img2.shape)
            if not is_valid:
                print(f"    ✗ HOMOGRAPHY VALIDATION FAILED: {reason}")
                return None
            
            print(f"    ✓ Homography validation passed")
            
            # Continue with actual stitching...
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            all_pts = np.concatenate([
                np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), 
                dst
            ])
            
            [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
            
            output_w = x_max - x_min
            output_h = y_max - y_min
            
            if output_w > 6000 or output_h > 6000:
                print(f"    ✗ OUTPUT TOO LARGE: {output_w}x{output_h}")
                return None
            
            print(f"    Output size: {output_w}x{output_h}")
            
            t = [-x_min, -y_min]
            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
            
            result = cv2.warpPerspective(img2, Ht.dot(M), (output_w, output_h))
            
            y1, y2 = t[1], min(h1 + t[1], output_h)
            x1, x2 = t[0], min(w1 + t[0], output_w)
            
            if y2 > y1 and x2 > x1:
                img1_region = img1[:y2-y1, :x2-x1]
                result_region = result[y1:y2, x1:x2]
                
                img1_mask = np.any(img1_region != 0, axis=2)
                result_mask = np.any(result_region != 0, axis=2)
                overlap_mask = img1_mask & result_mask
                
                if np.any(overlap_mask):
                    blended = self.enhanced_blend_with_seam(img1_region, result_region, img1_region, result_region, overlap_mask)
                    result[y1:y2, x1:x2] = blended
                else:
                    result[y1:y2, x1:x2] = img1_region
            
            print(f"    ✓ STITCHING SUCCESSFUL")
            return result
            
        except Exception as e:
            print(f"    ✗ EXCEPTION: {str(e)}")
            return None

    def post_process_final_image(self, img):
        """Post-process final stitched image to reduce artifacts"""
        if img is None:
            return None
        
        # 1. Remove small black holes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 2. Apply subtle denoising
        img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
        
        # 3. Enhance sharpness selectively
        gaussian = cv2.GaussianBlur(img_denoised, (0, 0), 2.0)
        img_sharp = cv2.addWeighted(img_denoised, 1.5, gaussian, -0.5, 0)
        
        # 4. Improve contrast slightly
        lab = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        return img_enhanced

    def process_complete_flight_stitching_with_turns(self):
        """Main processing function that includes turn sections in final area"""
        print("Starting Complete Improved GPS Flight Path Stitching (WITH TURNS)...")
        print("=" * 80)
        
        # Phase 1: Load and analyze flight data (unchanged)
        if not self.load_flight_data():
            print("Failed to load flight data")
            return False
        
        if not self.detect_turns_and_straight_sections():
            print("Failed to detect turns and straight sections")
            return False
        
        # Phase 2: Group sections (unchanged)
        straight_success = self.group_straight_sections()
        turn_success = self.group_turn_sections()
        
        if not straight_success and not turn_success:
            print("Failed to group any sections")
            return False
        
        # Phase 3: Stitch individual sections (unchanged)
        successful_straight_sections = 0
        section_results = {}
        
        if straight_success:
            print(f"\n{'='*20} STITCHING STRAIGHT SECTIONS {'='*20}")
            for section_data in self.straight_sections:
                result = self.stitch_section_with_subgroups(section_data, is_turn=False)
                if result is not None:
                    section_output = f"section_{section_data['section_id']:02d}_{section_data['direction']}_straight.jpg"
                    cv2.imwrite(section_output, result)
                    print(f"  ✓ Saved: {section_output}")
                    
                    # Store for final stitching
                    section_results[f"straight_{section_data['section_id']}"] = {
                        'type': 'straight',
                        'section_id': section_data['section_id'],
                        'direction': section_data['direction'],
                        'filepath': section_output,
                        'filename': section_output,
                        'image': result,
                        'start_seq': section_data['start_seq'],
                        'end_seq': section_data['end_seq'],
                        'center_lat': np.mean([img['lat'] for img in section_data['images']]),
                        'center_lon': np.mean([img['lon'] for img in section_data['images']])
                    }
                    successful_straight_sections += 1
                else:
                    print(f"  ✗ Section {section_data['section_id']} failed")
        
        successful_turn_sections = 0
        if turn_success:
            print(f"\n{'='*20} STITCHING TURN SECTIONS {'='*20}")
            for section_data in self.turn_sections:
                result = self.stitch_section_with_subgroups(section_data, is_turn=True)
                if result is not None:
                    section_output = f"turn_{section_data['section_id']:02d}_{section_data['turn_type']}.jpg"
                    cv2.imwrite(section_output, result)
                    print(f"  ✓ Saved: {section_output}")
                    
                    # Store for final stitching
                    section_results[f"turn_{section_data['section_id']}"] = {
                        'type': 'turn',
                        'section_id': section_data['section_id'],
                        'direction': 'TURN',
                        'filepath': section_output,
                        'filename': section_output,
                        'image': result,
                        'start_seq': section_data['start_seq'],
                        'end_seq': section_data['end_seq'],
                        'center_lat': section_data['center_lat'],
                        'center_lon': section_data['center_lon']
                    }
                    successful_turn_sections += 1
                else:
                    print(f"  ✗ Turn section {section_data['section_id']} failed")
        
        # Phase 4: Create final area image INCLUDING TURNS
        if len(section_results) > 1:
            print(f"\n{'='*20} CREATING FINAL AREA IMAGE (WITH TURNS) {'='*20}")
            
            # Convert section results to section_images format
            self.section_images = list(section_results.values())
            
            # Try enhanced spatial-aware stitching with turns
            final_result = self.stitch_sections_with_spatial_analysis_including_turns()
            
            if final_result is None:
                final_result = self.stitch_all_sections_sequential_improved()
            
            if final_result is not None:
                # Apply post-processing to final result
                final_result = self.post_process_final_image(final_result)
                
                output_filename = "complete_final_area_sift_with_turns.jpg"
                cv2.imwrite(output_filename, final_result)
                
                h, w = final_result.shape[:2]
                print(f"\n=== COMPLETE IMPROVED STITCHING SUMMARY (WITH TURNS) ===")
                print(f"Total images processed: {len(self.image_data)}")
                print(f"Straight sections successfully stitched: {successful_straight_sections}/{len(self.straight_sections) if straight_success else 0}")
                print(f"Turn sections successfully stitched: {successful_turn_sections}/{len(self.turn_sections) if turn_success else 0}")
                print(f"Final area image size: {w} x {h}")
                print(f"✓ Final area WITH TURNS saved: {output_filename}")
                print(f"\nAll outputs:")
                print(f"  Individual sections: section_*.jpg, turn_*.jpg")
                print(f"  Complete area with turns: {output_filename}")
                
                return True
            else:
                print("Failed to create final area image")
        
        print(f"\n=== SECTION STITCHING SUMMARY ===")
        print(f"Straight sections successfully stitched: {successful_straight_sections}")
        print(f"Turn sections successfully stitched: {successful_turn_sections}")
        print(f"Individual section files created")
        
        return (successful_straight_sections + successful_turn_sections) > 0
    
    def stitch_section_with_subgroups(self, section_data, is_turn=False):
        """Stitch section using sub-groups to reduce accumulation errors"""
        section_type = "Turn" if is_turn else "Straight"
        section_id = section_data['section_id']
        
        print(f"\n=== Stitching {section_type} Section {section_id} ===")
        print(f"Images: {section_data['image_count']}, Sequence: {section_data['start_seq']}-{section_data['end_seq']}")
        
        if section_data['image_count'] < 2:
            return None
        
        images = section_data['images']
        
        if len(images) <= 4:
            return self._stitch_direct(images)
        
        return self._stitch_with_subgroups(images)
    
    def _stitch_direct(self, images):
        """Direct stitching for small groups"""
        result = self.load_single_image(images[0])
        if result is None:
            return None
        
        successful_stitches = 1
        
        for i in range(1, len(images)):
            next_img = self.load_single_image(images[i])
            if next_img is None:
                continue
            
            stitched = self.stitch_two_images_improved_debug(result, next_img, images[i-1], images[i])
            
            if stitched is not None:
                result = stitched
                successful_stitches += 1
            
            del next_img
        
        return result if successful_stitches >= 2 else None
    
    def _stitch_with_subgroups(self, images):
        """Stitch using sub-groups to reduce error accumulation"""
        subgroup_size = 3
        overlap = 1
        subgroups = []
        
        for i in range(0, len(images), subgroup_size - overlap):
            end_idx = min(i + subgroup_size, len(images))
            if end_idx - i >= 2:
                subgroups.append(images[i:end_idx])
        
        subgroup_results = []
        for i, subgroup in enumerate(subgroups):
            sub_result = self.load_single_image(subgroup[0])
            if sub_result is None:
                continue
            
            sub_successful = 1
            for j in range(1, len(subgroup)):
                next_img = self.load_single_image(subgroup[j])
                if next_img is None:
                    continue
                
                stitched = self.stitch_two_images_improved_debug(sub_result, next_img, subgroup[j-1], subgroup[j])
                
                if stitched is not None:
                    sub_result = stitched
                    sub_successful += 1
                
                del next_img
            
            if sub_successful >= 2:
                subgroup_results.append((sub_result, subgroup[0], subgroup[-1]))
        
        if len(subgroup_results) < 2:
            return subgroup_results[0][0] if subgroup_results else None
        
        final_result = subgroup_results[0][0]
        
        for i in range(1, len(subgroup_results)):
            sub_img, first_img_data, last_img_data = subgroup_results[i]
            prev_first_data = subgroup_results[i-1][1]
            prev_last_data = subgroup_results[i-1][2]
            
            stitched = self.stitch_two_images_improved_debug(final_result, sub_img, prev_last_data, first_img_data)
            
            if stitched is not None:
                final_result = stitched
        
        return final_result
    
    # ===== FINAL AREA STITCHING METHODS =====
    
    def load_single_section_image(self, section_info, target_size=1200):
        """Load and resize a section image"""
        try:
            img = cv2.imread(section_info['filepath'])
            if img is None:
                return None
            
            h, w = img.shape[:2]
            ratio = min(target_size/w, target_size/h)
            if ratio < 1:
                new_w, new_h = int(w*ratio), int(h*ratio)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            return img
            
        except Exception as e:
            print(f"Error loading {section_info['filename']}: {e}")
            return None
    
    def validate_section_homography(self, M, img1_shape, img2_shape):
        """Validate homography transformation for section images"""
        if M is None:
            return False, "No homography matrix"
        
        if np.any(np.abs(M) > 2000):
            return False, "Extreme transformation values"
        
        h2, w2 = img2_shape[:2]
        corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        
        try:
            transformed_corners = cv2.perspectiveTransform(corners, M)
        except:
            return False, "Perspective transform failed"
        
        for corner in transformed_corners:
            x, y = corner[0]
            if abs(x) > 10000 or abs(y) > 10000:
                return False, "Extreme corner transformation"
        
        original_area = w2 * h2
        transformed_area = cv2.contourArea(transformed_corners)
        
        if transformed_area <= 0:
            return False, "Negative/zero transformed area"
        
        area_ratio = transformed_area / original_area
        if area_ratio < 0.05 or area_ratio > 20:
            return False, f"Excessive area distortion: {area_ratio:.2f}"
        
        return True, "Valid transformation"
    
    def stitch_two_section_images_improved(self, img1, img2, timeout_seconds=45):
        """Improved SIFT-based stitching of two section images"""
        start_time = time.time()
        
        try:
            sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=6, sigma=1.6)
            
            kp1, des1 = sift.detectAndCompute(img1, None)
            if time.time() - start_time > timeout_seconds or des1 is None or len(des1) < 15:
                return None
            
            kp2, des2 = sift.detectAndCompute(img2, None)
            if time.time() - start_time > timeout_seconds or des2 is None or len(des2) < 15:
                return None
            
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
            search_params = dict(checks=150)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(des1, des2, k=2)
            
            if time.time() - start_time > timeout_seconds:
                return None
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.65 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 12:
                return None
            
            # Geometric filtering for section images
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            displacements = pts2 - pts1
            median_displacement = np.median(displacements, axis=0)
            
            filtered_matches = []
            for i, match in enumerate(good_matches):
                displacement = displacements[i]
                diff_from_median = np.linalg.norm(displacement - median_displacement)
                
                if diff_from_median < 100:
                    filtered_matches.append(match)
            
            if len(filtered_matches) < 8:
                return None
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
            
            best_M = None
            best_inliers = 0
            
            for attempt in range(5):
                M, mask = cv2.findHomography(
                    dst_pts, src_pts, 
                    cv2.RANSAC, 
                    ransacReprojThreshold=3.0 + attempt * 1.0,
                    maxIters=8000,
                    confidence=0.99
                )
                
                if M is not None:
                    inliers = np.sum(mask) if mask is not None else 0
                    if inliers > best_inliers:
                        best_M = M
                        best_inliers = inliers
            
            M = best_M
            
            if M is None:
                return None
            
            is_valid, reason = self.validate_section_homography(M, img1.shape, img2.shape)
            if not is_valid:
                return None
            
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            all_pts = np.concatenate([
                np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2), 
                dst
            ])
            
            [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)
            
            output_w = x_max - x_min
            output_h = y_max - y_min
            
            if output_w > 20000 or output_h > 20000:
                return None
            
            t = [-x_min, -y_min]
            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
            
            result = cv2.warpPerspective(img2, Ht.dot(M), (output_w, output_h))
            
            # Advanced blending for overlap regions
            y1, y2 = t[1], min(h1 + t[1], output_h)
            x1, x2 = t[0], min(w1 + t[0], output_w)
            
            if y2 > y1 and x2 > x1:
                img1_region = img1[:y2-y1, :x2-x1]
                result_region = result[y1:y2, x1:x2]
                
                img1_mask = np.any(img1_region != 0, axis=2)
                result_mask = np.any(result_region != 0, axis=2)
                overlap_mask = img1_mask & result_mask
                
                if np.any(overlap_mask):
                    # Distance-based blending
                    dist_img1 = cv2.distanceTransform((~img1_mask).astype(np.uint8), cv2.DIST_L2, 5)
                    dist_result = cv2.distanceTransform((~result_mask).astype(np.uint8), cv2.DIST_L2, 5)
                    
                    total_dist = dist_img1 + dist_result + 1e-10
                    alpha = dist_result / total_dist
                    alpha = np.clip(alpha, 0, 1)
                    
                    for c in range(3):
                        blended_channel = (
                            alpha * img1_region[:, :, c] + 
                            (1 - alpha) * result_region[:, :, c]
                        )
                        
                        final_channel = np.where(
                            overlap_mask,
                            blended_channel,
                            np.where(img1_mask, img1_region[:, :, c], result_region[:, :, c])
                        )
                        
                        result[y1:y2, x1:x2, c] = final_channel.astype(np.uint8)
                else:
                    result[y1:y2, x1:x2] = img1_region
            
            return result
            
        except Exception as e:
            return None
    
    def analyze_section_spatial_relationships_with_turns(self):
        """Enhanced spatial analysis that includes turn sections"""
        print("\nAnalyzing spatial relationships between ALL sections (including turns)...")
        
        if len(self.section_images) < 2:
            return self.section_images
        
        # Sort all sections by sequence order for flight path reconstruction
        all_sections_by_sequence = sorted(self.section_images, key=lambda x: x['start_seq'])
        
        print("Flight sequence order:")
        for section in all_sections_by_sequence:
            print(f"  {section['type'].upper()} {section['section_id']}: seq {section['start_seq']}-{section['end_seq']}")
        
        # Group straight sections by direction (keep existing logic)
        straight_sections = [s for s in self.section_images if s['type'] == 'straight']
        turn_sections = [s for s in self.section_images if s['type'] == 'turn']
        
        direction_groups = {}
        for section in straight_sections:
            direction = section['direction']
            if direction not in direction_groups:
                direction_groups[direction] = []
            direction_groups[direction].append(section)
        
        for direction in direction_groups:
            direction_groups[direction].sort(key=lambda x: x['section_id'])
        
        opposite_directions = {
            'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E',
            'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'
        }
        
        stitching_plan = []
        processed_directions = set()
        
        # Create direction pairs for straight sections
        for direction, sections in direction_groups.items():
            if direction in processed_directions:
                continue
            
            opposite = opposite_directions.get(direction)
            if opposite in direction_groups and opposite not in processed_directions:
                stitching_plan.append({
                    'type': 'direction_pair',
                    'direction1': direction,
                    'sections1': sections,
                    'direction2': opposite,
                    'sections2': direction_groups[opposite]
                })
                processed_directions.add(direction)
                processed_directions.add(opposite)
            else:
                stitching_plan.append({
                    'type': 'single_direction',
                    'direction': direction,
                    'sections': sections
                })
                processed_directions.add(direction)
        
        # Add turn sections as separate stitching plan
        if turn_sections:
            stitching_plan.append({
                'type': 'turns_group',
                'sections': sorted(turn_sections, key=lambda x: x['start_seq'])
            })
        
        return stitching_plan
    
    def _stitch_turns_group(self, turn_sections):
        """Stitch turn sections together"""
        print(f"  Stitching {len(turn_sections)} turn sections")
        
        if len(turn_sections) == 0:
            return None
        
        if len(turn_sections) == 1:
            return self.load_single_section_image(turn_sections[0])
        
        # Sort by sequence to maintain flight order
        sorted_turns = sorted(turn_sections, key=lambda x: x['start_seq'])
        
        result = self.load_single_section_image(sorted_turns[0])
        if result is None:
            return None
        
        successful_stitches = 1
        
        for i in range(1, len(sorted_turns)):
            next_img = self.load_single_section_image(sorted_turns[i])
            if next_img is None:
                continue
            
            stitched = self.stitch_two_section_images_improved(result, next_img)
            
            if stitched is not None:
                result = stitched
                successful_stitches += 1
                print(f"    ✓ Turn {sorted_turns[i]['section_id']} added")
            else:
                print(f"    ✗ Turn {sorted_turns[i]['section_id']} failed")
            
            del next_img
        
        print(f"  Turn group: {successful_stitches}/{len(sorted_turns)} successful")
        return result if successful_stitches >= 1 else None
    
    def stitch_sections_with_spatial_analysis_including_turns(self):
        """Enhanced final stitching that includes turn sections"""
        print("\n=== Final Area Stitching with Spatial Analysis (INCLUDING TURNS) ===")
        
        if len(self.section_images) == 0:
            return None
        
        stitching_plan = self.analyze_section_spatial_relationships_with_turns()
        
        if not stitching_plan:
            return self.stitch_all_sections_sequential_improved()
        
        plan_results = []
        
        for i, plan in enumerate(stitching_plan):
            print(f"\nExecuting plan {i+1}/{len(stitching_plan)}: {plan['type']}")
            
            if plan['type'] == 'single_direction':
                result = self._stitch_single_direction(plan['sections'], plan['direction'])
                if result is not None:
                    plan_results.append(result)
                    print(f"  ✓ Single direction {plan['direction']} stitched")
            
            elif plan['type'] == 'direction_pair':
                result = self._stitch_direction_pair(
                    plan['sections1'], plan['direction1'],
                    plan['sections2'], plan['direction2']
                )
                if result is not None:
                    plan_results.append(result)
                    print(f"  ✓ Direction pair {plan['direction1']}+{plan['direction2']} stitched")
            
            elif plan['type'] == 'turns_group':
                result = self._stitch_turns_group(plan['sections'])
                if result is not None:
                    plan_results.append(result)
                    print(f"  ✓ Turn sections stitched")
        
        if len(plan_results) == 0:
            print("No plan results to combine")
            return None
        
        if len(plan_results) == 1:
            return plan_results[0]
        
        print(f"\nCombining {len(plan_results)} plan results...")
        final_result = plan_results[0]
        
        for i in range(1, len(plan_results)):
            print(f"  Combining result {i+1}...")
            stitched = self.stitch_two_section_images_improved(final_result, plan_results[i])
            
            if stitched is not None:
                final_result = stitched
                print(f"    ✓ Successfully combined")
            else:
                # Try reverse order
                stitched = self.stitch_two_section_images_improved(plan_results[i], final_result)
                if stitched is not None:
                    final_result = stitched
                    print(f"    ✓ Successfully combined (reversed)")
                else:
                    print(f"    ✗ Failed to combine, keeping previous result")
        
        return final_result
    
    def _stitch_single_direction(self, sections, direction):
        """Stitch sections within a single direction"""
        print(f"  Stitching {len(sections)} sections in direction {direction}")
        
        if len(sections) == 1:
            return self.load_single_section_image(sections[0])
        
        result = self.load_single_section_image(sections[0])
        if result is None:
            return None
        
        for i in range(1, len(sections)):
            next_img = self.load_single_section_image(sections[i])
            if next_img is None:
                continue
            
            stitched = self.stitch_two_section_images_improved(result, next_img)
            
            if stitched is not None:
                result = stitched
            
            del next_img
        
        return result
    
    def _stitch_direction_pair(self, sections1, direction1, sections2, direction2):
        """Stitch two complementary directions"""
        print(f"  Stitching direction pair: {direction1} + {direction2}")
        
        result1 = self._stitch_single_direction(sections1, direction1)
        result2 = self._stitch_single_direction(sections2, direction2)
        
        if result1 is None and result2 is None:
            return None
        elif result1 is None:
            return result2
        elif result2 is None:
            return result1
        
        combined = self.stitch_two_section_images_improved(result1, result2)
        
        if combined is not None:
            return combined
        else:
            combined = self.stitch_two_section_images_improved(result2, result1)
            if combined is not None:
                return combined
            else:
                h1, w1 = result1.shape[:2]
                h2, w2 = result2.shape[:2]
                return result1 if (h1 * w1) > (h2 * w2) else result2
    
    def stitch_all_sections_sequential_improved(self):
        """Improved sequential stitching as backup method"""
        print("\n=== Sequential Stitching (Backup) ===")
        
        if len(self.section_images) == 0:
            return None
        
        all_sections = sorted(self.section_images, key=lambda x: x['section_id'])
        
        result = self.load_single_section_image(all_sections[0])
        if result is None:
            return None
        
        for i in range(1, len(all_sections)):
            section = all_sections[i]
            next_img = self.load_single_section_image(section)
            if next_img is None:
                continue
            
            stitched = self.stitch_two_section_images_improved(result, next_img)
            
            if stitched is not None:
                result = stitched
            
            del next_img
        
        return result
                
    def process_complete_flight_stitching_with_turns(self):
        """Main processing function that includes turn sections in final area"""
        print("Starting Complete Improved GPS Flight Path Stitching (WITH TURNS)...")
        print("=" * 80)
        
        # Phase 1: Load and analyze flight data (unchanged)
        if not self.load_flight_data():
            print("Failed to load flight data")
            return False
        
        if not self.detect_turns_and_straight_sections():
            print("Failed to detect turns and straight sections")
            return False
        
        # Phase 2: Group sections (unchanged)
        straight_success = self.group_straight_sections()
        turn_success = self.group_turn_sections()
        
        if not straight_success and not turn_success:
            print("Failed to group any sections")
            return False
        
        # Phase 3: Stitch individual sections (unchanged)
        successful_straight_sections = 0
        section_results = {}
        
        if straight_success:
            print(f"\n{'='*20} STITCHING STRAIGHT SECTIONS {'='*20}")
            for section_data in self.straight_sections:
                result = self.stitch_section_with_subgroups(section_data, is_turn=False)
                if result is not None:
                    section_output = f"section_{section_data['section_id']:02d}_{section_data['direction']}_straight.jpg"
                    cv2.imwrite(section_output, result)
                    print(f"  ✓ Saved: {section_output}")
                    
                    # Store for final stitching
                    section_results[f"straight_{section_data['section_id']}"] = {
                        'type': 'straight',
                        'section_id': section_data['section_id'],
                        'direction': section_data['direction'],
                        'filepath': section_output,
                        'filename': section_output,
                        'image': result,
                        'start_seq': section_data['start_seq'],
                        'end_seq': section_data['end_seq'],
                        'center_lat': np.mean([img['lat'] for img in section_data['images']]),
                        'center_lon': np.mean([img['lon'] for img in section_data['images']])
                    }
                    successful_straight_sections += 1
                else:
                    print(f"  ✗ Section {section_data['section_id']} failed")
        
        successful_turn_sections = 0
        if turn_success:
            print(f"\n{'='*20} STITCHING TURN SECTIONS {'='*20}")
            for section_data in self.turn_sections:
                result = self.stitch_section_with_subgroups(section_data, is_turn=True)
                if result is not None:
                    section_output = f"turn_{section_data['section_id']:02d}_{section_data['turn_type']}.jpg"
                    cv2.imwrite(section_output, result)
                    print(f"  ✓ Saved: {section_output}")
                    
                    # Store for final stitching
                    section_results[f"turn_{section_data['section_id']}"] = {
                        'type': 'turn',
                        'section_id': section_data['section_id'],
                        'direction': 'TURN',
                        'filepath': section_output,
                        'filename': section_output,
                        'image': result,
                        'start_seq': section_data['start_seq'],
                        'end_seq': section_data['end_seq'],
                        'center_lat': section_data['center_lat'],
                        'center_lon': section_data['center_lon']
                    }
                    successful_turn_sections += 1
                else:
                    print(f"  ✗ Turn section {section_data['section_id']} failed")
        
        # Phase 4: Create final area image INCLUDING TURNS
        if len(section_results) > 1:
            print(f"\n{'='*20} CREATING FINAL AREA IMAGE (WITH TURNS) {'='*20}")
            
            # Convert section results to section_images format
            self.section_images = list(section_results.values())
            
            # Try enhanced spatial-aware stitching with turns
            final_result = self.stitch_sections_with_spatial_analysis_including_turns()
            
            if final_result is None:
                final_result = self.stitch_all_sections_sequential_improved()
            
            if final_result is not None:
                output_filename = "complete_final_area_sift_with_turns.jpg"
                cv2.imwrite(output_filename, final_result)
                
                h, w = final_result.shape[:2]
                print(f"\n=== COMPLETE IMPROVED STITCHING SUMMARY (WITH TURNS) ===")
                print(f"Total images processed: {len(self.image_data)}")
                print(f"Straight sections successfully stitched: {successful_straight_sections}/{len(self.straight_sections) if straight_success else 0}")
                print(f"Turn sections successfully stitched: {successful_turn_sections}/{len(self.turn_sections) if turn_success else 0}")
                print(f"Final area image size: {w} x {h}")
                print(f"✓ Final area WITH TURNS saved: {output_filename}")
                print(f"\nAll outputs:")
                print(f"  Individual sections: section_*.jpg, turn_*.jpg")
                print(f"  Complete area with turns: {output_filename}")
                
                return True
            else:
                print("Failed to create final area image")
        
        print(f"\n=== SECTION STITCHING SUMMARY ===")
        print(f"Straight sections successfully stitched: {successful_straight_sections}")
        print(f"Turn sections successfully stitched: {successful_turn_sections}")
        print(f"Individual section files created")
        
        return (successful_straight_sections + successful_turn_sections) > 0

def main():
    parser = argparse.ArgumentParser(description="Complete improved GPS flight path stitching")
    parser.add_argument("folder", help="Path to folder containing drone images")
    parser.add_argument("-o", "--output", default="complete_improved_flight", 
                       help="Output filename prefix")
    parser.add_argument("--include-turns", action="store_true", 
                       help="Include turn sections in final area image")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        return
    
    stitcher = CompleteImprovedGPSFlightPathStitcher(args.folder, args.output)
    
    stitcher.process_complete_flight_stitching_with_turns()
    

if __name__ == "__main__":
    main()