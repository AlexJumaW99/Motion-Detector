'''

1st frame is the bg which should be a static image - no mvmt 

2nd frame should be one with mvmt

3rd frame (Delta Frame) should be the diff between the 1st two frames,
the intensities of the pixels where the 2 images overlap is high and this shows mvmt,
while the intensity of the pixels where they don't are low and this shows no mvmt.

4th frame is the Threshold image, high intensity pixels will be coverted to white, showing mvmt,
while the low intensity pixels will be converted to black, showing no mvmt. 

Mvmt will only be considered if the high intensity pixels are > 500 and a rectangle will drawn surrounding them
IN THE ORIGINAL COLOR IMAGE as well as the time the object entered and exited the video. 

'''

#Import the modules 
import cv2 
from datetime import datetime
import pandas 

#Create a pandas df object to store the start and end times of mvmt 
df = pandas.DataFrame(columns=['Start','End'])

#Create a video object and access the webcam
video = cv2.VideoCapture(0) #add D_SHOW as a param if you run into issues

#Set the first frame to None for now, remember it should be static as it is the frame of reference
first_frame = None 
status_list = [None,None] #prevents a 'list index is out of range' error
times = [] #where we'll be storing the times when motion occurred

while True:
    #Read the frame 
    check, frame = video.read()

    status = 0 #meaning no movement 

    #Convert the frame to greyscale 
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Apply a Gaussian Blur for higher accuracy when it comes to motion detection
    #This is bc it smooths the image and does away with noise
    gray_frame = cv2.GaussianBlur(gray_frame,(21,21),0)

    #If the first frame is not there, then set it to the 1st frame recorded
    if first_frame is None: 
        first_frame = gray_frame
        continue #afterwards, go back to the start of the while Loop
    
    #Find the diff between the static 1st frame and the current frame 
    delta_frame = cv2.absdiff(first_frame,gray_frame)

    #Change the delta frame to a threshold frame to really distinguish the 1st and current frame 
    #Any pixel with an intensity higher than 30 should be made more intense (or lighter - 255) 
    #Any pixel with an intensity lower should be made black (darker - 0)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1] 

    #Remove the small and shadows 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    #Find the contours - high intensity - they show movement 
    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in cnts:
        if cv2.contourArea(contour) < 1000: #If they have a small area ignore them 
            continue 
        status = 1 #shows mvmt 

        (x, y, w, h) = cv2.boundingRect(contour) #get the coordinates of the rect vertices 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3) #draw the rect around the current colored frame 

    status_list.append(status) #append the status, either 1 or 0, to the status list 

    #Add the time the movement started to the times list 
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    
    #Add the time the movement ended 
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    #Display the frame 
    cv2.imshow('Grey Selfie', gray_frame)
    cv2.imshow('Delta Frame', delta_frame)
    cv2.imshow('Thresh Frame', thresh_frame)
    cv2.imshow('Final', frame)
    key = cv2.waitKey(2)

    if key == ord('q'):
        #Failsafe - in case the video is quit while mvmt was still being recorded, add the end time still 
        if status == 1:
            times.append(datetime.now())
        break
    
print(status_list)
print(times)

#Iterate through the times list with a STEP OF 2 and append the start and end times to the df 
for i in range(0, len(times), 2):
    df = df.append({'Start':times[i],'End':times[i+1]}, ignore_index=True)

#Export the data to a csv file 
df.to_csv('Times.csv')

#Release or turn off the webcam and destroy the window - this should happen when a user presses 'q' 
video.release()
cv2.destroyAllWindows()
