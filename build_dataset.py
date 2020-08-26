import cv2
import imutils
import numpy as np

bg = None


def run_avg(image, aWeight):
    global bg
    # background setup
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    # return nothing if no contours detected
    if len(cnts) == 0:
        return
    else:
        # find the largest contour which should be the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def main():
    # running average weight
    aWeight = 0.5

    # set up webcam
    camera = cv2.VideoCapture(0)

    # region of interest coordinates
    top, right, bottom, left = 10, 350, 225, 590

    num_frames = 0
    image_num = 0

    start_recording = False

    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()
        if (grabbed == True):

            # resize the frame
            frame = imutils.resize(frame, width=700)

            # flip and clone the frame
            frame = cv2.flip(frame, 1)
            clone = frame.copy()

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, wait for 30 frames before starting to detect the hand
            if num_frames < 30:
                run_avg(gray, aWeight)
                print(num_frames)
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(
                        clone, [segmented + (right, top)], -1, (0, 0, 255))
                    if start_recording:

                        # store the images in a directory as a png file
                        cv2.imwrite("Dataset/OneImages/one_" +
                                    str(image_num) + '.png', thresholded)
                        image_num += 1
                    cv2.imshow("Thesholded", thresholded)

            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q") or image_num > 1000:
                break

	    # if the user pressed "s", then start recording
            if keypress == ord("s"):
                start_recording = True

        else:
            print("Error")
            break


main()

camera.release()
cv2.destroyAllWindows()
