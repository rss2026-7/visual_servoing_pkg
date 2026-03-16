import cv2 as cv
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################



def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def preprocess(img, template):
    """
    Preprocess image to eliminate outliers and smooth corners
    """
    img = cv.GaussianBlur(img, (5,5), 0)

    return img


def get_mask(img):
    """
    Determine a mask of points with color with sufficiently close hue to hue
    """

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    low = np.array([5, 220, 100])
    high = np.array([30, 255, 255])

    mask = cv.inRange(hsv, low, high)

    return mask


def get_bounding_box(mask):
    """
    Find the bounding box of a mask
    Returns ((top left), (bottom right)) in the form x1,y1,x2,y2
    """
    
    
    contours, hiearchy = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    def rect_area(cnt):
        _,_,w,h = cv.boundingRect(cnt) 
        return w * h


    cnt = max(contours, key=rect_area)
    # find the largest contour

    x,y,w,h = cv.boundingRect(cnt) 

    debug_img = mask.copy()

    
    (x1, y1), (x2, y2) = ((x, y), (x + w, y + h))
    cv.rectangle(debug_img, (x1, y1), (x2, y2), (0,0,255), 2)
 
    # image_print(debug_img)

    return (x, y, x+w, y+h)



def cd_color_segmentation(img, template=None):
    """
    Implement the cone detection using color segmentation algorithm
    Input:
        img: np.3darray; the input image with a cone to be detected. BGR.
        template: Not required, but can optionally be used to automate setting hue filter values.
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
            (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
    """
    ########## YOUR CODE STARTS HERE ##########


    img = preprocess(img, template)

    mask = get_mask(img)

    # masked_img = cv.bitwise_and(img, img, mask=mask)
    
    # image_print(mask)
    # image_print(masked_img)

    x1,y1,x2,y2 = get_bounding_box(mask)

    bounding_box = ((x1, y1), (x2, y2))


    cv.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
 
    # image_print(debug_img)


    ########### YOUR CODE ENDS HERE ###########

    # Return bounding box
    return bounding_box
