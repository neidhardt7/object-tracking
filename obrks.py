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

# Function to handle key events
def key_listener(event, x, y, flags, param):
    global paused
    if event == cv2.EVENT_LBUTTONDOWN:
        paused = not paused

def start_slideshow(cap):
    global paused
    #"""
    # Initialize window
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
            
        key = cv2.waitKey(5)  # Change delay to adjust speed of slideshow
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
    #"""
    return None

def track_object(cap, fps):
    global paused
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Starting object tracking...")

    # Initialize the object tracker
    tracker = cv2.legacy.TrackerMOSSE_create() #cv2.TrackerKCF_create()

    success, frame = cap.read()
    if not success:
        print("Failed to read initial frame for tracking.")
        return False

    # Get the initial bounding box of the object
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
    #bbox=[top left x, top left y, width, height]
    
    origin = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    # Initialize displacement values
    tracker.init(frame, bbox)
    print("Origin: (",origin[0],",",origin[1],")")
    #print("Origin: (",0,",",0,")")
    cv2.destroyWindow('Select Object')

    # Timer starts
    tvid = 0
    arr = []
    cp = []

    while True:
        # Read a new frame
        success, frame = cap.read()
        if not success:
            print("Failed to read frame during tracking.")
            break
        
        # Update the tracker
        success, box = tracker.update(frame)

        # Draw the bounding box and calculate displacement        
        if success:
            x, y, w, h = [int(v) for v in box]
            #print(x,y,w,h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 130, 0), 2)
            #The (B,G,R) is the intensity value of blue, green and red
            current_position = (x + (w / 2), y + (h / 2))
            pos = (int(x + (w / 2)), int(y + (h / 2)))
            #cv2.circle(frame,(int(x + (w/2)), int(y + (h/2))), 100, (240,130,0), 2)
            xn = round(current_position[0] - origin[0], 6)
            yn = round(-current_position[1] - origin[1], 6)
            Coordinates = (xn, yn, round(tvid, 6))

            # Display the displacement value on the frame
            cv2.putText(frame, f"Coordinates: {Coordinates}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # Display Coordinates
            print("(x, y, t):", {Coordinates})

            for i in range(1, len(cp)):
                if cp[i - 1] is None or cp[i] is None:
                    continue
                cv2.line(frame, cp[i - 1], cp[i], (0, 0, 255), 2)

            cp.append(np.array(pos))
            arr.append(np.array(Coordinates))

            # Display the frame
            cv2.imshow("Object Tracking", frame)

            # Pause video if 'p' is pressed
            key = cv2.waitKey(1)
            if key == ord('p'):
                paused = not paused
            if key == ord('d'):
                return False
            # Exit if 'q' is pressed
            if key == ord('q'):
                break
            tvid = tvid + (1 / fps)
        else:
                print("Object lost. Pausing tracking...")
                break

    cv2.destroyWindow('Object Tracking')
    x, y, t = np.array(arr).T

    #slopes
    slopex, cx, rx, px, std_errx = stats.linregress(t, x)
    slopey, cy, ry, py, std_erry = stats.linregress(t, y)
    print("Slope of x vs t :",round(slopex,6))
    print("Slope of y vs t :",round(slopey,6))
    print()

    #conversion factor
    rw=21.95 #26.12 #21.8 avij #new 25.4,21.95
    rh=16.46 #19.6 #16.4 avij #new 19.05,16.46
    mx=rw/width
    my=rh/height

    print("Video dimensions :",width,"px X ",height,"px")
    print("Real-world dimensions :",rw,"mm X ",rh,"mm")
    print()

    #velocity
    velx = round(slopex * mx, 6)
    vely = round(slopey * my, 6)
    vel = round(math.sqrt(velx ** 2 + vely ** 2), 6)
    print("x-velocity :", velx, " mm/s", ", y-velocity :", vely, " mm/s")
    print("velocity of drop is :", vel, " mm/s")

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
    #cv2.imshow("Path",frame)
    str=folder_path+"/vel.tsv"
    fname=os.path.basename(filename)
    with open(str, "ab") as f:
        np.savetxt(f,np.column_stack(([fname], [vel])),delimiter='\t',fmt='%s')

    return True

for filename in glob.glob(os.path.join(folder_path, '*.mp4')):
    # Initialize the video capture
    cap = cv2.VideoCapture(filename) #carslow,car2slow,car3slow,drop1h,'drop3.mp4' for vid
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #cap.set(cv2.CAP_DSHOW,0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps=cap.get(cv2.CAP_PROP_FPS)
    #fps=fps*2

    while True:
        frame = start_slideshow(cap)
        if frame is None:
            break
        print("Slideshow paused. Starting tracking...")
        if not track_object(cap, fps):
            break
        #continue_prompt = input("Do you want to track the next object? (y/n): ")
        #if continue_prompt.lower() != 'y':
        #    break
        paused = False

cv2.destroyAllWindows()

