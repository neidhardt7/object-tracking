import os,glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy import stats
import math
#new program
folder_path = "C:/Users/rachi/AppData/Local/Programs/Python/Python313/FFBatch" #os.getcwd(), convert backslash to frontflash

def select_multiple_objects(frame):
    ROIs = cv2.selectROIs("Select Objects", frame, fromCenter=False)
    cv2.destroyWindow('Select Objects')
    return ROIs

# Define colors for different trajectories
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

for filename in glob.glob(os.path.join(folder_path, '*.mp4')):
  with open(filename, 'r') as f:
        # Initialize the video capture
        cap = cv2.VideoCapture(filename)  #carslow,car2slow,car3slow,drop1h,'drop3.mp4' for vid
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #cap.set(cv2.CAP_DSHOW,0)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps=cap.get(cv2.CAP_PROP_FPS)
        #fps=fps*2
        
        success, frame = cap.read()
        if not success:
            continue

        paused = False
        speed = 1
        def key_listener(event, x, y, flags, param):
            global paused
            if event == cv2.EVENT_LBUTTONDOWN:
                paused = not paused
        cv2.namedWindow('Slideshow')
        cv2.setMouseCallback('Slideshow', key_listener)

        while True:
            if not paused:
                success, frame = cap.read()
                if not success:
                    break
                cv2.imshow('Slideshow', frame)
                key = cv2.waitKey(10) # Change delay to adjust speed of slideshow
                if key == ord('q'):
                 paused = not paused
                 cv2.destroyWindow('Slideshow')
                 break
                elif key == ord('f'):
                 speed = min(speed + 100, 110)  # Fast forward, speed up by increasing frame skip
                elif key == ord('r'):
                 cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2 * fps))  # Rewind    

        ROIs = select_multiple_objects(frame)
        trackers = []
        origins = []
        trajectories = [[] for _ in range(len(ROIs))]

        for roi in ROIs:
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, tuple(roi))
            trackers.append(tracker)
            origin = ((roi[0] + roi[2] / 2), (roi[1] + roi[3] / 2))
            origins.append(origin)

        print(f"Number of objects = {len(trackers)}")

        tvid = 0
        arr = [[] for _ in range(len(trackers))]
        total_distances = [0 for _ in range(len(trackers))]
        while True:
            # Read a new frame
            success, frame = cap.read()
            if not success:
                break

            for i, tracker in enumerate(trackers):
                # Update the tracker
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #The (B,G,R) is the intensity value of blue, green and red
                    current_position = (x + (w / 2), y + (h / 2))
                    pos = (int(x + (w/2)), int(y + (h/2)))
                    xn = round(current_position[0] - origins[i][0], 6)
                    yn = round(-current_position[1] - origins[i][1], 6)
                    #pos = (int(x+w/2),int(y+h/2))
                    #current_position=pos
                    Coordinates = (xn, yn, round(tvid, 6))

                    if len(trajectories[i]) > 0:
                        prev_position = trajectories[i][-1]
                        distance = math.sqrt((current_position[0] - prev_position[0])**2 + (current_position[1] - prev_position[1])**2)
                        total_distances[i] += distance

                    # Display the displacement value on the frame
                    cv2.putText(frame, f"Box {i+1} Coordinates: {Coordinates}", (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    arr[i].append(np.array(Coordinates))
                    # Display Coordinates
                    #print(f"Object {i+1} (x, y, t):", {Coordinates})
                    
                    # Update trajectory
                    trajectories[i].append(current_position)
                    for j in range(1, len(trajectories[i])):
                        cv2.line(frame, (int(trajectories[i][j-1][0]), int(trajectories[i][j-1][1])),
                                 (int(trajectories[i][j][0]), int(trajectories[i][j][1])),
                                 colors[i % len(colors)], 2)
            # Display the frame
            cv2.imshow("Object Tracking", frame)

            # Pause video if 'p' is pressed
            key = cv2.waitKey(1)
            if key == ord('p'):
             paused = not paused
            # Exit if 'q' is pressed
            elif key == ord('q'):
             break
            tvid=tvid+(1/fps) 

        cv2.destroyWindow('Object Tracking')

        for i, coords in enumerate(arr):
            x, y, t = np.array(coords).T
            #dists=np.array(arr2)
            #ldist=dists[0]
            #print(ldist)
            #slopes
            slopex, _, _, _, _ = stats.linregress(t, x)
            slopey, _, _, _, _ = stats.linregress(t, y)
            print("Slope of x vs t :",round(slopex,6))
            print("Slope of y vs t :",round(slopey,6))
            #conversion factor
            rw=190.5#11.77 #26.12 #21.8 avij
            rh=107.2#6.67 #19.6 #16.4 avij
            mx=rw/width
            my=rh/height

            #velocity
            velx = round(slopex * mx, 6)
            vely = round(slopey * my, 6)
            vel = round(math.sqrt(velx ** 2 + vely ** 2), 6)
            print(f"Box {i+1} velocity: {vel} mm/s, time of motion of box {i+1}: {tvid} s")
            print(f"Object {i+1} total distance travelled: {total_distances[i]} mm")
            """
	    # Create subplots
	    fig, axs = plt.subplots(1, 3, figsize=(5, 15))

	    # Plot y vs t
	    axs[0].plot(t, y, 'b-')
	    axs[0].set_title('y vs t')
	    axs[0].set_xlabel('t')
	    axs[0].set_ylabel('y')
	    
	    # Plot x vs t
	    axs[1].plot(t, x, 'r-')
	    axs[1].set_title('x vs t')
	    axs[1].set_xlabel('t')
	    axs[1].set_ylabel('x')
	    
	    # Plot y vs x
	    axs[2].plot(x,y, 'g-')
	    axs[2].set_title('y vs x')
	    axs[2].set_xlabel('x')
	    axs[2].set_ylabel('y')
	    """
            cv2.imshow("Path",frame)
            #im_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #plt.show()
            print()
            str_file = folder_path + "/dist.tsv"
            fname = os.path.basename(filename) + f" Box {i+1}"
            with open(str_file, "ab") as f:
                np.savetxt(f, np.column_stack(([fname], [total_distances[i]])),delimiter='\t', fmt='%s')
                
        #cv2.destroyWindow('Object Tracking')
        # Release video capture and close windows
        #cap.release()
        cv2.destroyAllWindows()
