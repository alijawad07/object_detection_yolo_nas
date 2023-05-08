import os
import cv2
import torch
import argparse
import yaml
import time
from super_gradients.common.object_names import Models
from super_gradients.training import models
from tqdm import tqdm
import shutil

# Define a function to generate a unique color for each class label
def get_color(label):
    hash_value = hash(str(label))
    # Add a large constant value to handle negative hash values
    color = [int(x) for x in str((hash_value + 2**32) % (256**3))][0:3]
    return tuple(color)

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--data', required=True, help='Path to YAML config file')
    parser.add_argument('--source', required=True, help='Path to directory containing images')
    parser.add_argument('--output', required=True, help='Path to save the inference result')
    parser.add_argument('--weights', help='Path to checkpoint file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for bounding boxes')
    parser.add_argument('--video', action='store_true', help='Flag indicating whether the source directory contains videos instead of images')
    return parser.parse_args()

args = parse_args()

# Load dataset parameters from YAML config file
with open(args.data, 'r') as f:
    config = yaml.safe_load(f)
classes = config['names']

# Load the best model checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.weights:

    best_model = models.get(
        'yolo_nas_l',
        num_classes=len(classes),
        checkpoint_path=args.weights
    )
else:
    best_model = models.get("yolo_nas_l", pretrained_weights="coco")

best_model.eval()
best_model.to(device)

# Set up output directories
output_dir_images = os.path.join(args.output, 'images')
output_dir_labels = os.path.join(args.output, 'labels')

if os.path.exists(output_dir_images):
    shutil.rmtree(output_dir_images)

if os.path.exists(output_dir_labels):
    shutil.rmtree(output_dir_labels)

os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_labels, exist_ok=True)

label_colors = {}
for label in classes:
    label_colors[label] = get_color(label)

label_names = {i: label for i, label in enumerate(classes)}

# Loop over all image files in the directory
for filename in tqdm(os.listdir(args.source)):
    if args.video and filename.endswith('.mp4'):
        # Read the video file
        video_path = os.path.join(args.source, filename)
        cap = cv2.VideoCapture(video_path)

        # Get the video frame rate and dimensions
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the output video writer
        out_video = cv2.VideoWriter(os.path.join(args.output, filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Path to the output txt file for this video
        output_file = os.path.join(output_dir_labels, os.path.splitext(filename)[0] + '.txt')

        # Open the output file for writing
        with open(output_file, 'w') as f:
            frame_count = 0

            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform inference
                predictions = best_model.predict(frame)

                boxes = []
                scores = []
                labels_int = []
                height, width, _ = frame.shape
                # Write the detections to the output file in YOLO format
                for pred in predictions:
                    bboxes_xyxy = pred.prediction.bboxes_xyxy
                    confidence = pred.prediction.confidence
                    labels = pred.prediction.labels

                    # Draw bounding box only if the confidence is above the threshold
                    for i in range(len(bboxes_xyxy)):
                        if confidence[i] >= args.conf:
                            
                            label = classes[int(labels[i])]
                            color = get_color(label)
                            #color = (0, 0, 255)
                            thickness = 2

                            # Draw the bounding box and label
                            x1, y1, x2, y2 = [int(coord) for coord in bboxes_xyxy[i]]

                            boxes.append([x1, y1, x2, y2])
                            scores.append(float(confidence[i]))
                            labels_int.append(int(label))


                # Draw the bounding boxes and class labels on the image
                for box, score, label in zip(boxes, scores, labels_int):
                    
                    x1, y1, x2, y2 = box
                    color = get_color(label)
                    class_name = label_names.get(label, 'unknown')
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
                    cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                      
                
                # Convert the image to PNG format
                filename =  f'{time.time()}.jpg'

                # Save the image with bounding boxes to the output directory
                output_path = os.path.join(output_dir_images, filename)
                cv2.imwrite(output_path, frame)

                for i in range(len(bboxes_xyxy)):
                    x_center = (bboxes_xyxy[i][0] + bboxes_xyxy[i][2]) / 2 / frame.shape[1]
                    y_center = (bboxes_xyxy[i][1] + bboxes_xyxy[i][3]) / 2 / frame.shape[0]
                    width = (bboxes_xyxy[i][2] - bboxes_xyxy[i][0]) / frame.shape[1]
                    height = (bboxes_xyxy[i][3] - bboxes_xyxy[i][1]) / frame.shape[0]
                    f.write(f"{int(labels[i])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                frame_count += 1

                # Write the frame to the output video
                out_video.write(frame)

            cap.release()
            out_video.release()

    else:
               
        # Read the image
        image_path = os.path.join(args.source, filename)
        image = cv2.imread(image_path)

        # Perform inference
        predictions = best_model.predict(image)

        # Path to the output txt file for this image
        output_file = os.path.join(output_dir_labels, os.path.splitext(filename)[0] + '.txt')

        # Open the output file for writing
        with open(output_file, 'w') as f:
            boxes = []
            scores = []
            labels_int = []
            
            # Write the detections to the output file in YOLO format
            for pred in predictions:
                bboxes_xyxy = pred.prediction.bboxes_xyxy
                confidence = pred.prediction.confidence
                labels = pred.prediction.labels

                # Draw bounding box only if the confidence is above the threshold
                for i in range(len(bboxes_xyxy)):
                    if confidence[i] >= args.conf:
                        
                        label = classes[int(labels[i])]
                        color = get_color(label)
                        #color = (0, 0, 255)
                        thickness = 2

                        # Draw the bounding box and label
                        x1, y1, x2, y2 = [int(coord) for coord in bboxes_xyxy[i]]

                        boxes.append([x1, y1, x2, y2])
                        scores.append(float(confidence[i]))
                        labels_int.append(label)


            # Draw the bounding boxes and class labels on the image
            for box, score, label in zip(boxes, scores, labels_int):
                
                x1, y1, x2, y2 = box
               
                color = get_color(label)
                label_int = None
                for k, v in label_names.items():
                    if v == label:
                        label_int = k
                        break
                class_name = label_names.get(label_int, 'unknown')
               
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
                cv2.putText(image, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            # Save the image with bounding boxes to the output directory
            output_path = os.path.join(output_dir_images, filename)
            cv2.imwrite(output_path, image)

            for i in range(len(bboxes_xyxy)):
                x_center = (bboxes_xyxy[i][0] + bboxes_xyxy[i][2]) / 2 / image.shape[1]
                y_center = (bboxes_xyxy[i][1] + bboxes_xyxy[i][3]) / 2 / image.shape[0]
                width = (bboxes_xyxy[i][2] - bboxes_xyxy[i][0]) / image.shape[1]
                height = (bboxes_xyxy[i][3] - bboxes_xyxy[i][1]) / image.shape[0]
                f.write(f"{int(labels[i])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Release resources
cv2.destroyAllWindows()
