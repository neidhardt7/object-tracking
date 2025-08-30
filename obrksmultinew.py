import os
import glob
import cv2
import numpy as np
import math
from scipy import stats
#new program
# Get the folder path and convert backslashes
folder_path = "C:/Users/rachi/AppData/Local/Programs/Python/Python313/FFBatch" #os.getcwd(), convert backslash to frontflash

# Global pause variable
paused = False

# Define colors for different trajectories
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

# Function to handle key events
def key_listener(event, x, y, flags, param):
    global paused
    if event == cv2.EVENT_LBUTTONDOWN:
        paused = not paused

def select_multiple_objects(frame):
    """Improved ROI selection function"""
    ROIs = cv2.selectROIs("Select Objects", frame, fromCenter=False)
    cv2.destroyWindow('Select Objects')
    return ROIs

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def track_objects(cap, fps):
    global paused
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Starting multiple object tracking...")

    # Real-world dimensions (in mm)
    rw = 190.5  # Real width in mm
    rh = 107.2  # Real height in mm
    mx = rw/width   # mm per pixel in x direction
    my = rh/height  # mm per pixel in y direction

    print("Video dimensions:", width, "px X", height, "px")
    print("Real-world dimensions:", rw, "mm X", rh, "mm")
    print("Conversion factors: x =", mx, "mm/px, y =", my, "mm/px")

    # Read first frame
    success, frame = cap.read()
    if not success:
        print("Failed to read initial frame for tracking.")
        return False

    # Select multiple ROIs using improved function
    print("Select objects to track and press Enter")
    print("Press 'q' to finish selection")
    bboxes = select_multiple_objects(frame)
    
    if len(bboxes) == 0:
        print("No objects selected")
        return False

    # Initialize trackers and tracking data
    trackers = []
    origins = []
    total_distances_px = [0.0] * len(bboxes)  # Distance in pixels
    total_distances_mm = [0.0] * len(bboxes)  # Distance in millimeters
    previous_positions = []
    paths = [[] for _ in range(len(bboxes))]
    tracking_times = [0.0] * len(bboxes)  # Time each object is tracked
    coordinates = [[] for _ in range(len(bboxes))]  # Store coordinates for velocity calculation
    
    # Initialize trackers and get origins
    for bbox in bboxes:
        tracker = cv2.legacy.TrackerKCF_create()
        tracker.init(frame, bbox)
        trackers.append(tracker)
        
        # Calculate origin (center of initial bounding box)
        origin = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        origins.append(origin)
        previous_positions.append(origin)
        print(f"Origin: ({origin[0]}, {origin[1]})")

    # Timer starts
    tvid = 0
    start_time = cv2.getTickCount()  # Start timing

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame during tracking.")
            break
        
        # Calculate elapsed time
        current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        current_positions = []
        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in box]
                center = (x + w//2, y + h//2)
                current_positions.append(center)
                tracking_times[i] = current_time  # Update tracking time
                
                # Draw bounding box with object-specific color
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i % len(colors)], 2)
                
                # Calculate and update distance
                if previous_positions[i] is not None:
                    # Calculate distance in pixels
                    distance_px = calculate_distance(previous_positions[i], center)
                    total_distances_px[i] += distance_px
                    
                    # Convert to millimeters
                    dx = abs(center[0] - previous_positions[i][0]) * mx
                    dy = abs(center[1] - previous_positions[i][1]) * my
                    distance_mm = math.sqrt(dx**2 + dy**2)
                    total_distances_mm[i] += distance_mm
                
                # Store path and coordinates
                paths[i].append(center)
                xn = round(center[0] - origins[i][0], 6)
                yn = round(-center[1] - origins[i][1], 6)
                coordinates[i].append((xn, yn, round(tvid, 6)))
                
                # Draw path with object-specific color
                for j in range(1, len(paths[i])):
                    cv2.line(frame, paths[i][j-1], paths[i][j], colors[i % len(colors)], 2)
                
                # Display distance and time
                cv2.putText(frame, f"Dist: {total_distances_mm[i]:.1f} mm", 
                          (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)
                cv2.putText(frame, f"Time: {current_time:.1f} s", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)
            else:
                current_positions.append(None)
                cv2.putText(frame, f"Object {i+1}: Tracking lost", 
                          (100, 80 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        previous_positions = current_positions
        
        # Display frame with total elapsed time
        cv2.putText(frame, f"Total Time: {current_time:.1f} s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Multiple Object Tracking", frame)
        
        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord('p'):
            paused = not paused
        if key == ord('d'):
            return False
        if key == ord('q'):
            break
            
        tvid += 1/fps

    cv2.destroyWindow('Multiple Object Tracking')
    
    # Print final results and calculate velocities
    print("\nFinal tracking results:")
    for i, (dist_px, dist_mm, time, coords) in enumerate(zip(total_distances_px, total_distances_mm, tracking_times, coordinates)):
        print(f"Object {i+1}:")
        print(f"  Distance: {dist_px:.2f} pixels ({dist_mm:.2f} mm)")
        print(f"  Time tracked: {time:.2f} seconds")
        
        # Calculate velocities using linear regression
        if len(coords) > 1:
            x, y, t = np.array(coords).T
            slopex, _, _, _, _ = stats.linregress(t, x)
            slopey, _, _, _, _ = stats.linregress(t, y)
            velx = round(slopex * mx, 6)
            vely = round(slopey * my, 6)
            vel = round(math.sqrt(velx**2 + vely**2), 6)
            print(f"  Velocity: {vel:.2f} mm/s (x: {velx:.2f}, y: {vely:.2f})")
    
    # Save results to file with improved format
    str = folder_path + "/Distance.tsv"
    fname = os.path.basename(filename)
    with open(str, "ab") as f:
        # Save filename, distances (mm), times, velocities, and coordinates
        for i, (dist_mm, time, coords) in enumerate(zip(total_distances_mm, tracking_times, coordinates)):
            if len(coords) > 1:
                x, y, t = np.array(coords).T
                slopex, _, _, _, _ = stats.linregress(t, x)
                slopey, _, _, _, _ = stats.linregress(t, y)
                velx = round(slopex * mx, 6)
                vely = round(slopey * my, 6)
                vel = round(math.sqrt(velx**2 + vely**2), 6)
                data_row = [f"{fname} Object {i+1}", dist_mm, time, vel, velx, vely]
                np.savetxt(f, [data_row], delimiter='\t', fmt='%s')

    return True

def start_slideshow(cap):
    global paused
    cv2.namedWindow('Slideshow')
    paused = False
    speed = 1
    
    cv2.setMouseCallback('Slideshow', key_listener)
    
    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame during slideshow.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return None
            
            cv2.imshow('Slideshow', frame)
            
        key = cv2.waitKey(200)  # Change delay to adjust speed of slideshow
        if key == ord('q'):
            paused = not paused
            return frame
        elif key == ord('f'):
          speed = min(speed + 100, 110)  # Fast forward, speed up by increasing frame skip
        elif key == ord('r'):
          cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2 * fps))  # Rewind    
        elif key == ord('d'):
            cv2.destroyWindow('Slideshow')
            return None
    
    return None

for filename in glob.glob(os.path.join(folder_path, '*.mp4')):
    # Initialize the video capture
    cap = cv2.VideoCapture(filename)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        frame = start_slideshow(cap)
        if frame is None:
            break
        print("Slideshow paused. Starting tracking...")
        if not track_objects(cap, fps):
            break
        paused = False

cv2.destroyAllWindows()

