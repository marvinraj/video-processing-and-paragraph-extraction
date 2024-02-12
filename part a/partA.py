import cv2
import os
import numpy as np

# Initialize the variable to track whether it is day or night
is_day = False

def day_night_checker(video_path, threshold_brightness=100, brightness_factor=2.0):
    
    global is_day
    
    # Print the information about checking the video
    print("Checking", video_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_brightness = 0
    
    # Loop through all frames in the video
    for frame_count in range(total_frames):
        # Read the frame from the video
        ret, frame = cap.read()
        # Check if there are no more frames
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the brightness of the frame
        brightness = np.mean(gray_frame)
        total_brightness += brightness

    # Release video capture
    cap.release()

    # Calculate the average brightness over all frames
    average_brightness = total_brightness / total_frames if total_frames > 0 else 0
    
    # Check if it's nighttime based on average brightness
    is_night = average_brightness < threshold_brightness

    # Print the output information
    if is_night:
        print(f"The video is taken during nighttime with an average brightness of {average_brightness}.")
        print(f"Commencing brightening process...")
        
        # Update the day/night variable
        is_day = False
        
        # Open the video file again
        cap = cv2.VideoCapture(video_path)

        # Set up video writer for the brightened video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = "brightened_" + video_path 
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

        # Loop through all frames in the video and increase brightness
        for frame_count in range(total_frames):
            # Read the frame from the video
            ret, frame = cap.read()
            # Check if there are no more frames
            if not ret:
                break

            # Increase the brightness of the frame
            brightened_frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

            # Write the frame to the output video
            out.write(brightened_frame)
            
        # Release video capture and writer for the brightened video
        cap.release()
        out.release()

        print(f"Brightened video saved at: {output_path}\n")
    else:
        # Update the day/night variable
        is_day = True
        print(f"The video is taken during daytime with an average brightness of {average_brightness}.")
        print("Video will NOT be brightened.\n")

    

# This functions blurs the video and outputs it
def blur_video(name, outputName="blurred"):
    
    # Open the input video file
    cap = cv2.VideoCapture(name)

    # Load a pre-trained face detector for face blurring
    face_detector = cv2.CascadeClassifier("face_detector.xml")

    # Get the total number of frames in the video
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Create a VideoWriter object for the output video
    out = cv2.VideoWriter(outputName + ".avi",cv2.VideoWriter_fourcc(*'MJPG'),30.0,(1280, 720))

    # Iterate through all frames in the video
    for frame_count in range(0, int(total_frames)):
        # Read the current frame from the input video
        retval, frame = cap.read()
        
        # Resize the frame to a standard size (1280x720 pixels)
        frame = cv2.resize(frame, (1280, 720))
        
        # Increase the overall brightness of the frame by adding a constant value
        frame = cv2.add(frame, 10)
        
        # Convert the frame to grayscale for face detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame using a pre-trained face detector
        detections = face_detector.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)

        # Apply Gaussian blur to the regions containing detected faces
        for (x, y, w, h) in detections:
            frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (15, 15), cv2.BORDER_DEFAULT)
        
        # Write the processed frame to the output video
        out.write(frame)
        
    # Release the resources for the input video and the output video
    cap.release()
    out.release()



# This function overlays video 2 onto video 1 and saves the output
def overlay_video(video_1, video_2, outputName="overlay"):
    
    # Open the first and second input video file respectively
    cap_video1 = cv2.VideoCapture(video_1)
    cap_video2 = cv2.VideoCapture(video_2)

    # Get the total number of frames in each input video
    total_frames_video1 = cap_video1.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames_video2 = cap_video2.get(cv2.CAP_PROP_FRAME_COUNT)

    # Create a VideoWriter object for the output video
    out = cv2.VideoWriter(outputName + ".avi", cv2.VideoWriter_fourcc(*'MJPG'),30.0, (1280, 720))

    # Iterate through all frames, taking the maximum frame count of the two videos
    for frame_count in range(0, int(max(total_frames_video1, total_frames_video2))):
        # Read a frame from the first and second input video respectively
        ret1, frame_video1 = cap_video1.read()
        ret2, frame_video2 = cap_video2.read()
        
        # Check if there are no more frames in the first video
        if not ret1:
            break
        
        # Define the position and size for overlaying the second video onto the first
        x_offset = 15
        y_offset = 15
        width = 300
        height = 220
        
        # Check if there are more frames in the second video
        if ret2:
            # Resize the second video to the specified dimensions            
            resize = cv2.resize(frame_video2, (width, height))
            
            # Overlay the resized second video onto the first video
            frame_video1[y_offset:y_offset + height, x_offset:x_offset + width] = resize
            
            # Draw a black border around the overlay region
            startPt = (x_offset, y_offset)
            endPt = (x_offset + width, y_offset + height)
            black = (0, 0, 0)
            thickness = 2
            frame_video1 = cv2.rectangle(frame_video1, startPt, endPt, black, thickness)

        # Write the processed frame to the output video
        out.write(frame_video1)
        
    # Release the resources for the input video and the output video
    cap_video1.release()
    cap_video2.release()
    out.release()



# This function adds watermarks to the video frames
def add_watermark(video, outputName="after_watermark"):
    # Open the input video file
    cap_video = cv2.VideoCapture(video)
    
    # Get the total number of frames in the input video
    total_frame = cap_video.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Create a VideoWriter object for the output video
    out = cv2.VideoWriter(outputName + ".avi",cv2.VideoWriter_fourcc(*'MJPG'),30.0,(1280, 720))

    # Load watermark images
    watermark_1 = cv2.imread("watermark1.png")
    watermark_2 = cv2.imread("watermark2.png")
    
    # Set the initial watermark to watermark_1
    current_watermark = watermark_1
    
    # Boolean variable to switch between watermark_1 and watermark_2
    switch_watermark = True
    
    # Iterate through all frames in the input video
    for frame_count in range(0, int(total_frame)):
        # Check if it's time to switch watermarks (every 30 frames)
        if frame_count % 30 == 0:
            switch_watermark = not switch_watermark
            # Set the current watermark based on the switch state
            if switch_watermark:
                current_watermark = watermark_1
            else:
                current_watermark = watermark_2
                
        # Read a frame from the input video
        ret, frame = cap_video.read()
        
        # Add the current watermark to the frame
        frame = cv2.add(frame, current_watermark)
        
        # Write the frame with the watermark to the output video
        out.write(frame)
        
    # Release the resources for the input video and the output video
    cap_video.release()
    out.release()

# This function adds an endscreen to the final video
def add_endscreen(final_video, endscreen, outputName="final_with_endscreen"):
    # Open the input final video file and endscreen video file respectively
    cap_final = cv2.VideoCapture(final_video)
    cap_endscreen = cv2.VideoCapture(endscreen)
    
    # Get the total number of frames in the final video and endscreen
    total_frame_final = cap_final.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frame_endscreen = cap_endscreen.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Create a VideoWriter object for the output video
    out = cv2.VideoWriter(outputName + ".avi", cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (1280, 720))

    # Read and write frames from the final video
    for frame_count in range(int(total_frame_final)):
        ret, frame = cap_final.read()
        
        # Check if there are no more frames in the final video
        if not ret:
            break
        # Write the frame from the final video to the output video
        out.write(frame)

    # Read and write frames from the endscreen
    for frame_count in range(int(total_frame_endscreen)):
        ret_endscreen, frame_endscreen = cap_endscreen.read()
        # Check if there are no more frames in the endscreen
        if not ret_endscreen:
            break
        # Write the frame from the endscreen to the output video
        out.write(frame_endscreen)

    # Release the resources for the input video and the output video
    cap_final.release()
    cap_endscreen.release()
    out.release()


# List of videos to be processed
video_to_process = [ "office", "singapore","traffic"]
# File name of the endscreen video
endscreen_video = "endscreen.mp4"

# Loop through each video in the list for processing
for x in video_to_process:
    # Print the name of the video being processed
    print("\nVideo to process: " + x + "\n")
    
    # Step 1: Brightening Video
    if not os.path.isfile("brightened_" + x + ".avi"):
        print("Step 1 : Brightening Video")
        print("Checking if the video is day or night...")
        day_night_checker(x + ".mp4")
        
    # Step 2: Blurring Video
    if not os.path.isfile("blurred_" + x + ".avi"):
        print("Step 2 : Blurring Video")
        print("Video blurring in progress...")
        blur_video(x + ".mp4" if is_day else "brightened_" + x + ".mp4", outputName="blurred_" + x)
        print("Blurring Video Completed.\n")
        
    # Step 3: Overlaying Video
    if not os.path.isfile("overlay_" + x + ".avi"):
        print("Step 3 : Overlaying Video")
        print("Video overlaying in progress...")
        overlay_video("blurred_" + x + ".avi", "talking.mp4", outputName="overlay_" + x)
        print("Overlaying Video Completed.\n")
        
    # Step 4: Applying Watermark
    if not os.path.isfile("after_watermark_" + x + ".avi"):
        print("Step 4 : Applying Watermark")
        print("Applying watermark to video in progress...")
        add_watermark("overlay_" + x + ".avi", outputName="after_watermark_" + x)
        print("Applying Watermark Completed.\n")
        
    # Step 5: Adding Endscreen
    if not os.path.isfile("final_video_" + x + ".avi"):
        print("Step 5 : Adding Endscreen")
        print("Adding endscreen to video in progress...")
        add_endscreen("after_watermark_" + x + ".avi",endscreen_video, outputName="final_video_" + x)
        print("Adding Endscreen Completed.\n")

    # Print a message indicating the completion of processing for the current video
    print("Video Processed: " + x + "\n")