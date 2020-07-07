import cv2
import matplotlib.pyplot as plt


def load_img(filename):
    """load image in rgb format from given filename"""
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_bboxes_on_image(filename, bboxes):
    """plot bounding boxes on given image"""
    img = load_img(filename)
    img_copy = img.copy()
    for i in range(len(bboxes)):
        x1, y1, x2, y2  = bboxes[i]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_copy    
    
def show_bboxes(filename, bboxes, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    
    img = plot_bboxes_on_image(filename, bboxes)
    plt.imshow(img)
    plt.title("Image Id: {}, Number of bboxes: {}".format(filename.split('/')[-1], len(bboxes)))
    plt.show()