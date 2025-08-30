import os
import glob
import cv2
import numpy as np
import math
from scipy import stats
from collections import defaultdict

# Get the folder path and convert backslashes
folder_path = "C:/Users/rachi/AppData/Local/Programs/Python/Python313/FFBatch"

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
    total_distances_px = [0.0] * len(bboxes)
    total_distances_mm = [0.0] * len(bboxes)
    previous_positions = []
    paths = [[] for _ in range(len(bboxes))]
    tracking_times = [0.0] * len(bboxes)
    coordinates = [[] for _ in range(len(bboxes))]
    
    # New: Dictionary to store displacement at time intervals
    time_displacement = [defaultdict(list) for _ in range(len(bboxes))]
    time_interval = 1.0  # Record displacement every 1 second
    
    # New: List to store relative separation between first two objects
    relative_separation = []
    
    # Initialize trackers and get origins
    for bbox in bboxes:
        tracker = cv2.legacy.TrackerKCF_create()
        tracker.init(frame, bbox)
        trackers.append(tracker)
        
        origin = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        origins.append(origin)
        previous_positions.append(origin)
        print(f"Origin: ({origin[0]}, {origin[1]})")

    # Timer starts
    tvid = 0
    start_time = cv2.getTickCount()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame during tracking.")
            break
        
        current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        current_positions = []
        for i, tracker in enumerate(trackers):
            success, box = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in box]
                center = (x + w//2, y + h//2)
                current_positions.append(center)
                tracking_times[i] = current_time
                
                # Draw bounding box with object-specific color
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i % len(colors)], 2)
                
                # Calculate and update distance
                if previous_positions[i] is not None:
                    distance_px = calculate_distance(previous_positions[i], center)
                    total_distances_px[i] += distance_px
                    
                    dx = abs(center[0] - previous_positions[i][0]) * mx
                    dy = abs(center[1] - previous_positions[i][1]) * my
                    distance_mm = math.sqrt(dx**2 + dy**2)
                    total_distances_mm[i] += distance_mm
                    
                    # New: Record displacement at time intervals
                    current_interval = math.floor(current_time / time_interval) * time_interval
                    if current_interval > 0:  # Skip 0 second interval
                        time_displacement[i][current_interval].append(total_distances_mm[i])
                
                # Store path and coordinates
                paths[i].append(center)
                xn = round(center[0] - origins[i][0], 6)
                yn = round(-center[1] - origins[i][1], 6)
                coordinates[i].append((xn, yn, round(tvid, 6)))
                
                # Draw path and display info
                for j in range(1, len(paths[i])):
                    cv2.line(frame, paths[i][j-1], paths[i][j], colors[i % len(colors)], 2)
                
                cv2.putText(frame, f"Dist: {total_distances_mm[i]:.1f} mm", 
                          (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)
                cv2.putText(frame, f"Time: {current_time:.1f} s", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 2)
            else:
                current_positions.append(None)
                cv2.putText(frame, f"Object {i+1}: Tracking lost", 
                          (100, 80 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        previous_positions = current_positions
        
        # Record relative separation if at least two objects are tracked and both are detected in this frame
        if len(current_positions) >= 2 and current_positions[0] is not None and current_positions[1] is not None:
            # Calculate separation in pixels
            sep_px = calculate_distance(current_positions[0], current_positions[1])
            # Convert to mm (using average of mx and my for isotropic scaling)
            sep_mm = math.sqrt(((current_positions[0][0] - current_positions[1][0]) * mx) ** 2 + ((current_positions[0][1] - current_positions[1][1]) * my) ** 2)
            relative_separation.append((current_time, sep_mm))
        
        cv2.putText(frame, f"Total Time: {current_time:.1f} s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Multiple Object Tracking", frame)
        
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
        
        if len(coords) > 1:
            x, y, t = np.array(coords).T
            slopex, _, _, _, _ = stats.linregress(t, x)
            slopey, _, _, _, _ = stats.linregress(t, y)
            velx = round(slopex * mx, 6)
            vely = round(slopey * my, 6)
            vel = round(math.sqrt(velx**2 + vely**2), 6)
            print(f"  Velocity: {vel:.2f} mm/s (x: {velx:.2f}, y: {vely:.2f})")
    cv2.imshow("Path",frame)
    # Save results to files
    str_distance = folder_path + "/Distance.tsv"
    str_displacement = folder_path + "/Displacement_Time.tsv"
    fname = os.path.basename(filename)
    
    # Save original distance data
    with open(str_distance, "ab") as f:
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
    
    # New: Save displacement over time data
    with open(str_displacement, "ab") as f:
        for i, displacement_data in enumerate(time_displacement):
            for time_interval, displacements in sorted(displacement_data.items()):
                if displacements:  # Only save if we have data for this interval
                    avg_displacement = sum(displacements) / len(displacements)
                    data_row = [f"{fname} Object {i+1}", time_interval, avg_displacement]
                    np.savetxt(f, [data_row], delimiter='\t', fmt='%s')

    # Save relative separation data if available
    if len(relative_separation) > 0:
        rel_sep_path = os.path.join(folder_path, "Relative_Separation.tsv")
        with open(rel_sep_path, "w") as f:
            f.write("Time(s)\tSeparation(mm)\n")
            for t, sep in relative_separation:
                f.write(f"{t:.3f}\t{sep:.3f}\n")
        print(f"Relative separation data saved to {rel_sep_path}")

        # Plot the relative separation data
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            df_sep = pd.read_csv(rel_sep_path, sep='\t')
            plt.figure(figsize=(10, 6))
            plt.plot(df_sep['Time(s)'], df_sep['Separation(mm)'], marker='o', color='purple')
            plt.xlabel('Time (s)')
            plt.ylabel('Separation (mm)')
            plt.title('Relative Separation Between Two Objects Over Time')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plot_path = os.path.join(folder_path, 'Relative_Separation.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Relative separation plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not plot relative separation: {e}")

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
            
        key = cv2.waitKey(200)
        if key == ord('q'):
            paused = not paused
            return frame
        elif key == ord('f'):
          speed = min(speed + 100, 110)
        elif key == ord('r'):
          cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2 * fps))
        elif key == ord('d'):
            cv2.destroyWindow('Slideshow')
            return None
    
    return None

for filename in glob.glob(os.path.join(folder_path, '*.mp4')):
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
