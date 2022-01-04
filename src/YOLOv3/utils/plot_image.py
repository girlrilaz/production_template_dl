import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox
import tensorflow as tf

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def save_images(display_list):

    # Create a new image that includes the bounding boxes
    output_image = draw_bbox(img, bbox, label, conf)

    # Create output folder for the images with the bounding boxes
    dir_name = "images_with_boxes"
    if not os.path.exists(os.path.join(data_config, dir_name)):
        os.mkdir(os.path.join(data_config, dir_name))
    
    # Save the image in the directory images_with_boxes
    cv2.imwrite(os.path.join(data_config, 'images_with_boxes', filename), output_image)
    
    # Display the image with bounding boxes
    # display(Image(f'images_with_boxes/{filename}'))
    display(Image(os.path.join(data_config, 'images_with_boxes', filename)))
