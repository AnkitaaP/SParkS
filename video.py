# test_video.py
#
# Open a video input file and feed each image frame to 'openalpr'
# for license plate recognition.

import numpy as np
import cv2
import webcolors
from openalpr import Alpr


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def main():
    writer = None
    VIDEO_SOURCE = '/home/ankita/Downloads/video1.mp4'
    WINDOW_NAME = 'openalpr'
    FRAME_SKIP = 15
    alpr = Alpr('us', 'us.conf', '/usr/local/share/openalpr/runtime_data')
    if not alpr.is_loaded():
        print('Error loading OpenALPR')
        # sys.exit(1)
    alpr.set_top_n(3)
    # alpr.set_default_region('new')

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        alpr.unload()
        print('Failed to load video file')
        #sys.exit('Failed to open video file!')
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowTitle(WINDOW_NAME, 'OpenALPR video test')

    _frame_number = 0

    currentFrame = 0

    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter('color_output.avi', fourcc, 2.0, (int(width), int(height)))

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            print('VidepCapture.read() failed. Exiting...')
            break

        _frame_number += 1
        if _frame_number % FRAME_SKIP != 0:
            continue

        results = alpr.recognize_ndarray(frame)

        for i, plate in enumerate(results['results']):
            list1 = []
            best_candidate = plate['candidates'][0]
            min_coord = plate['coordinates'][0]
            max_coord = plate['coordinates'][2]
            x_min = int(min_coord['x'])
            y_min = int(min_coord['y'])
            x_max = int(max_coord['x'])
            y_max = int(max_coord['y'])
            cv2.resize(frame, (500, 500), fx=0, fy=0,
                       interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (255, 255, 0), 5)
            #cropimg = cv2.getRectSubPix(
             #  frame, (x_min-10, y_min), (x_min, y_min))
            #type(cropimg[0:])
            #x = cropimg[0:][0][0]
            #list1 = x.tolist()
            #list1 = list(map(int,list1))
            #print(list1)
            #print(type(list1))
            #act,pred = get_colour_name(tuple(list1))
            #print(pred)
            
            cv2.putText(frame, 'NP: '+str(best_candidate['plate']).upper(), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.85, [255, 255, 0], 2)
            if(best_candidate['confidence'] > 50):
                print('Plate #{}: {:7s} ({:.2f}%)'.format(i, best_candidate['plate'].upper(), best_candidate['confidence']))
            list1 = []
            cv2.imshow(WINDOW_NAME, frame)

            
            out.write(frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    alpr.unload()


if __name__ == "__main__":
    main()
