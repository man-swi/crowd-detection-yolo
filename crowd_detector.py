import cv2
import numpy as np
from ultralytics import YOLO
import csv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# --- Configuration Constants ---
VIDEO_PATH = r'D:\AI\Crowd detection\production_id_4196258 (720p).mp4' 
OUTPUT_CSV_PATH = "crowd_log.csv"
OUTPUT_VIDEO_PATH = "output_video_crowd_detection.mp4" 

MODEL_NAME = 'yolov8s.pt' 
PERSON_CLASS_ID = 0      # COCO class ID for 'person'

# Crowd Definition Parameters
MIN_PERSONS_IN_GROUP = 3            # Minimum number of persons to be considered a group
MAX_DISTANCE_WITHIN_GROUP = 120     # Max distance (pixels) between person centroids to form/belong to a group
MIN_CONSECUTIVE_FRAMES_FOR_CROWD = 10 # Min frames a group must persist to be classified as a crowd
MAX_DISTANCE_GROUP_TRACKING = 300   # Max distance (pixels) a group's centroid can move between frames for tracking
MAX_FRAMES_TO_MISS_GROUP = 4        # Grace period (frames) a tracked group can be undetected before being considered lost

# Detection and Output Options
DETECTION_CONFIDENCE = 0.25         # Minimum confidence for YOLO person detection
DRAW_VISUALIZATIONS = True          # Display video with annotations during processing
SAVE_OUTPUT_VIDEO = True            # Save annotated video to OUTPUT_VIDEO_PATH


def get_centroid(bbox):
    """Calculates the centroid (center point) of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def calculate_group_centroid(group_bboxes):
    """Calculates the overall centroid of a group of bounding boxes."""
    if not group_bboxes:
        return None
    person_centroids_x = [get_centroid(b)[0] for b in group_bboxes]
    person_centroids_y = [get_centroid(b)[1] for b in group_bboxes]
    return int(np.mean(person_centroids_x)), int(np.mean(person_centroids_y))

def detect_persons_in_frame(frame, yolo_model, person_cls_id, min_conf):
    """Detects persons in a frame using YOLO."""
    detected_bboxes = []
 
    results = yolo_model(frame, stream=True, verbose=False, classes=[person_cls_id], conf=min_conf)
    for res in results: 
        for box in res.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            detected_bboxes.append([x1, y1, x2, y2])
    return detected_bboxes

def find_groups_of_close_persons(person_bboxes, max_dist, min_persons):
    """
    Identifies groups of closely located persons using connected components.
    Persons are nodes; an edge exists if their centroids are within max_dist.
    """
    num_persons = len(person_bboxes)
    if not person_bboxes or num_persons < min_persons: 
        return []
    
    centroids = [get_centroid(bbox) for bbox in person_bboxes]
    
    dist_matrix = cdist(np.array(centroids), np.array(centroids))
    adj_matrix = dist_matrix < max_dist
    
    visited = [False] * num_persons
    all_groups_indices = [] # Stores lists of indices (relative to person_bboxes) for each group

    for i in range(num_persons):
        if not visited[i]:
            current_group_person_indices = []
            queue = [i] 
            visited[i] = True
            head = 0 
            while head < len(queue): 
                person_idx_u = queue[head]; head += 1
                current_group_person_indices.append(person_idx_u)
                for person_idx_v in range(num_persons): # Check adjacency with all other persons
                    if adj_matrix[person_idx_u, person_idx_v] and not visited[person_idx_v]:
                        visited[person_idx_v] = True
                        queue.append(person_idx_v)
            all_groups_indices.append(current_group_person_indices)
    
    # Filter groups by minimum number of persons
    valid_groups_bboxes_list = []
    for group_indices in all_groups_indices:
        if len(group_indices) >= min_persons:
            group_bboxes = [person_bboxes[idx] for idx in group_indices]
            valid_groups_bboxes_list.append(group_bboxes)
            
    return valid_groups_bboxes_list

# --- Main Application Logic ---
def run_crowd_detection():
    """
    Main function to run the crowd detection process on a video.
    It follows these general steps:
    1. Initialize model, video input/output, CSV.
    2. For each frame:
        a. Detect persons (YOLO).
        b. Group close persons.
        c. Track groups across frames (Hungarian algorithm).
        d. Identify persistent groups as crowds.
        e. Log crowd events to CSV.
        f. Visualize (optional) and save output video (optional).
    3. Clean up resources.
    """
    print("Initializing Crowd Detection System...")
    print(f"  Input Video: {VIDEO_PATH}")
    print(f"  YOLO Model: {MODEL_NAME} (Confidence: {DETECTION_CONFIDENCE})")
    print(f"  Crowd Definition: {MIN_PERSONS_IN_GROUP}+ persons, within {MAX_DISTANCE_WITHIN_GROUP}px, for {MIN_CONSECUTIVE_FRAMES_FOR_CROWD}+ frames.")
    print(f"  Tracking: Max group centroid move {MAX_DISTANCE_GROUP_TRACKING}px, Grace period {MAX_FRAMES_TO_MISS_GROUP} frames.")

    cap = None # Initialize cap to None for robust error handling
    out_video = None 

    try:
        
        try:
            model = YOLO(MODEL_NAME)
            print(f"YOLO model '{MODEL_NAME}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}. Exiting.")
            print("Ensure 'ultralytics' is installed (pip install ultralytics) and the model file is accessible.")
            return

        # Open video
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {VIDEO_PATH}. Exiting.")
            return
        
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0: 
            print(f"Warning: Video FPS reported as {video_fps}, defaulting to 25.0 FPS for output video.")
            video_fps = 25.0 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video loaded: {frame_w}x{frame_h} @ {video_fps:.2f} FPS, {total_frames if total_frames > 0 else 'Unknown number of'} frames.")

        # Open CSV file for writing using 'with' for automatic closing
        with open(OUTPUT_CSV_PATH, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Frame Number", "Person Count in Crowd", "Group ID"]) # Added Group ID

            # Initialize VideoWriter for output video (optional)
            if SAVE_OUTPUT_VIDEO:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_fps, (frame_w, frame_h))
                    print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")
                except Exception as e:
                    print(f"Error opening VideoWriter: {e}. Output video will not be saved.")
                 
            
            tracked_groups = []         # List of dictionaries for tracked groups
            next_group_id_counter = 0   # Assigns unique IDs to new groups
            frame_number = 0            # Frame counter

            print("\nStarting video processing...")
           
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if total_frames > 0 and frame_number < total_frames:
                         print(f"Warning: Error reading frame {frame_number + 1}, but not at end of reported total frames.")
                    else:
                         print("End of video stream reached.")
                    break
                
                frame_number += 1
                # Progress update
                if frame_number == 1 or \
                   (video_fps > 0 and frame_number % int(video_fps * 5) == 0) or \
                   (video_fps <= 0 and frame_number % 100 == 0):
                     progress_info = f"  Processing Frame: {frame_number}"
                     if total_frames > 0: progress_info += f" / {total_frames}"
                     print(progress_info)

                # --- Step 2: Person Detection in a Single Frame ---
                person_bboxes = detect_persons_in_frame(frame, model, PERSON_CLASS_ID, DETECTION_CONFIDENCE)

                # --- Step 3: Grouping Close Persons in a Single Frame ---
                current_frame_potential_groups_bboxes = find_groups_of_close_persons(
                    person_bboxes, MAX_DISTANCE_WITHIN_GROUP, MIN_PERSONS_IN_GROUP
                )

                # Prepare data for current frame's groups for tracking
                current_groups_data_for_tracking = []
                for group_bb_list in current_frame_potential_groups_bboxes:
                    group_centroid = calculate_group_centroid(group_bb_list)
                    if group_centroid:
                        current_groups_data_for_tracking.append({
                            'bboxes': group_bb_list, 
                            'centroid': group_centroid, 
                            'person_count': len(group_bb_list), 
                            'matched_to_tracker': False # Flag for matching logic
                        })
                
                for tg_dict in tracked_groups: tg_dict['updated_this_frame'] = False

                # --- Step 4: Tracking Groups Across Frames ---
                if tracked_groups and current_groups_data_for_tracking:
                    candidate_tracked_centroids = np.array([tg['centroid'] for tg in tracked_groups])
                    candidate_current_centroids = np.array([cg['centroid'] for cg in current_groups_data_for_tracking])

                    if candidate_tracked_centroids.size > 0 and candidate_current_centroids.size > 0:
                        cost_matrix = cdist(candidate_current_centroids, candidate_tracked_centroids)
                     
                        current_group_indices, matched_tracked_group_indices = linear_sum_assignment(cost_matrix)

                        for cg_idx, tg_idx in zip(current_group_indices, matched_tracked_group_indices):
                            cost = cost_matrix[cg_idx, tg_idx]
                            if cost < MAX_DISTANCE_GROUP_TRACKING:
                                current_group_info = current_groups_data_for_tracking[cg_idx]
                                tracked_groups[tg_idx].update({
                                    'bboxes': current_group_info['bboxes'], 
                                    'centroid': current_group_info['centroid'],
                                    'person_count': current_group_info['person_count'],
                                    'consecutive_frames': tracked_groups[tg_idx]['consecutive_frames'] + 1,
                                    'last_seen_frame': frame_number,
                                    'frames_since_last_seen': 0,
                                    'updated_this_frame': True
                                })
                                current_groups_data_for_tracking[cg_idx]['matched_to_tracker'] = True
                
                # Add new, unmatched current groups to tracked_groups
                for cg_data in current_groups_data_for_tracking:
                    if not cg_data['matched_to_tracker']:
                        next_group_id_counter += 1
                        tracked_groups.append({
                            'id': next_group_id_counter, 
                            'bboxes': cg_data['bboxes'], 
                            'centroid': cg_data['centroid'],
                            'person_count': cg_data['person_count'], 
                            'consecutive_frames': 1,
                            'last_seen_frame': frame_number, 
                            'frames_since_last_seen': 0,
                            'updated_this_frame': True,
                            'printed_crowd_alert': False
                        })

                # Manage lost groups (apply grace period)
                surviving_tracked_groups = []
                for tg_dict in tracked_groups:
                    if not tg_dict['updated_this_frame']:
                        tg_dict['frames_since_last_seen'] = frame_number - tg_dict['last_seen_frame']
                    
                    if tg_dict['frames_since_last_seen'] <= MAX_FRAMES_TO_MISS_GROUP:
                        surviving_tracked_groups.append(tg_dict)
                    # else: 
                        # print(f"  Group ID {tg_dict['id']} (P:{tg_dict['person_count']}, F:{tg_dict['consecutive_frames']}) LOST (missed {tg_dict['frames_since_last_seen']}).")
                tracked_groups = surviving_tracked_groups
                
                # Reset printed_crowd_alert flag if a group is no longer a crowd
                for group in tracked_groups:
                    if group['consecutive_frames'] < MIN_CONSECUTIVE_FRAMES_FOR_CROWD:
                        group['printed_crowd_alert'] = False 
                
                # --- Step 5: Crowd Detection and Logging ---
                # --- Step 6: Output and Visualization ---
                frame_to_draw_on = frame.copy()
                current_frame_has_crowd_event = False

                # Draw all initially detected persons (thin green)
                for p_bbox in person_bboxes:
                    cv2.rectangle(frame_to_draw_on, (p_bbox[0], p_bbox[1]), (p_bbox[2], p_bbox[3]), (0, 255, 0), 1)

                # Define colors for different tracked groups for visualization (BGR format)
                group_viz_colors = [(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(128,0,128),(0,128,128),(128,128,0)]
                
                for tg_data in tracked_groups:
                    group_color = group_viz_colors[tg_data['id'] % len(group_viz_colors)]
                    # Highlight persons in this tracked group
                    for person_box_in_group in tg_data['bboxes']:
                        cv2.rectangle(frame_to_draw_on, (person_box_in_group[0],person_box_in_group[1]), 
                                      (person_box_in_group[2],person_box_in_group[3]), group_color, 2) # Thickness 2
                    
                    # Prepare and display group info text
                    info_text_parts = [f"ID:{tg_data['id']}", f"P:{tg_data['person_count']}", f"F:{tg_data['consecutive_frames']}"]
                    if not tg_data['updated_this_frame'] and tg_data['frames_since_last_seen'] > 0:
                        info_text_parts.append(f"M:{tg_data['frames_since_last_seen']}")
                    info_text = " ".join(info_text_parts)
                    
                    (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_bg_y1 = tg_data['centroid'][1] - 7 - text_h # Position text above centroid
                    # Ensure text background doesn't go off-screen (simple check for top)
                    text_bg_y1 = max(text_bg_y1, 0)
                    text_y = max(tg_data['centroid'][1] - 7, text_h)

                    cv2.rectangle(frame_to_draw_on, (tg_data['centroid'][0], text_bg_y1), 
                                  (tg_data['centroid'][0] + text_w, text_y), (220,220,220), -1) # Light gray BG
                    cv2.putText(frame_to_draw_on, info_text, (tg_data['centroid'][0], text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, group_color, 1, cv2.LINE_AA)

                    # Check for crowd, log to CSV, and annotate "CROWD" text
                    if tg_data['consecutive_frames'] >= MIN_CONSECUTIVE_FRAMES_FOR_CROWD:
                        current_frame_has_crowd_event = True
                        csv_writer.writerow([frame_number, tg_data['person_count'], tg_data['id']])
                        
                        # Print to console less frequently for detected crowds (e.g., once per second of video)
                        print_interval = int(video_fps) if video_fps > 0 else 25 # Default to 25 if fps is 0
                        if not tg_data.get('printed_crowd_alert', False) or tg_data['consecutive_frames'] % print_interval == 0:
                            print(f"  >>> CROWD DETECTED: Group ID {tg_data['id']} ({tg_data['person_count']}p) for {tg_data['consecutive_frames']} frames. Logged.")
                            tg_data['printed_crowd_alert'] = True
                        
                        # "CROWD" text near group centroid
                        crowd_text_y = tg_data['centroid'][1] + 20
                        cv2.putText(frame_to_draw_on, "CROWD", (tg_data['centroid'][0], crowd_text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA) # Black outline
                        cv2.putText(frame_to_draw_on, "CROWD", (tg_data['centroid'][0], crowd_text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA) # Yellow text

                # Display overall frame number and crowd status on the frame
                cv2.putText(frame_to_draw_on, f"Frame: {frame_number}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                if current_frame_has_crowd_event:
                    cv2.putText(frame_to_draw_on, "CROWD EVENT(S) ACTIVE", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA) # Red

                # Save the annotated frame to the output video file
                if SAVE_OUTPUT_VIDEO and out_video is not None:
                    out_video.write(frame_to_draw_on)

                # Display the processed frame (optional)
                if DRAW_VISUALIZATIONS:
                    display_width = 1280 
                    if frame_w > 0 and frame_h > 0 :
                        aspect_ratio = frame_h / frame_w
                        display_height = int(display_width * aspect_ratio)
                        if display_height > 0 and display_width > 0 :
                            resized_frame_for_display = cv2.resize(frame_to_draw_on, (display_width, display_height), interpolation=cv2.INTER_AREA)
                            cv2.imshow("Crowd Detection System", resized_frame_for_display)
                        else: 
                            cv2.imshow("Crowd Detection System", frame_to_draw_on) # Fallback
                    else: 
                         cv2.imshow("Crowd Detection System", frame_to_draw_on) # Fallback

                    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                        print("Quit signal received. Stopping video processing.")
                        break
            
            # --- End of Main Loop --- Cleanup is handled after the 'with' block or in 'finally'

    except IOError as e: # Handles error opening/writing CSV file
        print(f"Error with CSV file '{OUTPUT_CSV_PATH}': {e}. Exiting.")
    except Exception as e: # Catch any other unexpected errors during processing
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Step 7: Close video and CSV file (Cleanup) ---
        print("\nVideo processing finished or stopped. Cleaning up resources...")
        if cap is not None and cap.isOpened():
            cap.release()
            print("Video capture released.")
        
        # CSV file is closed automatically by 'with' statement or if IOError occurred before opening
        print(f"CSV file operations for '{OUTPUT_CSV_PATH}' handled.")
        
        if SAVE_OUTPUT_VIDEO and out_video is not None: # out_video might exist even if not isOpened() if creation failed mid-way
            if out_video.isOpened(): # Check if it was successfully opened before trying to release
                 out_video.release()
            print(f"Output video operations for '{OUTPUT_VIDEO_PATH}' handled.")
        
        if DRAW_VISUALIZATIONS:
            cv2.destroyAllWindows()
            print("OpenCV display windows closed.")
        
        print("\nCrowd detection program finished.")


if __name__ == "__main__":
    run_crowd_detection()
