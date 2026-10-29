import numpy as np
from PIL import ImageGrab
import cv2
import time
from direction_keys import ReleaseKey, PressKey, W, A, S, D


def draw_lines(img,lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except:
        pass

def process_img(orig_img):
    '''
    This function will take a orignal image and perform Canny edge detection on it.
    Inputs: nd.array of orignal image; "orig_img"
    Output: nd.array of processed image' "proc_img"

    '''
    proc_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    proc_img = cv2.Canny(proc_img, threshold1=200, threshold2=300)
    
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32)
    #proc_img = roi(proc_img, [vertices])

    lines = cv2.HoughLinesP(proc_img, 1, np.pi/180, 180, 20, 15)
    draw_lines(proc_img,lines)

    return proc_img


def roi(img, vertices):
    '''
    This function makes an np.zeros array and masks it on the original image.
    We can define which region is to be masked
    '''

    # This is masking with opencv
    #blank mask:
    mask = np.zeros_like(img)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked


    '''
    Please Ignore this for now...
    something i will try later
    total_rows, total_cols, layers = img.shape
    print(img.shape)
    X, Y = np.ogrid[:total_rows, :total_cols]
    '''


def screen_grab(): 
    while True:
        
        #Grab the screen only the region we want
        #PS: We'll be monitoring many other things simultaneously so the game won't be displayed
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,800)))            #This is the actual screen
        new_screen = process_img(screen)                                   #This the processed image; we need this processed image for finding lanes
        cv2.imshow('window', new_screen)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_grab()
