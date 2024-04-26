# CNN-for-object-detection
edge-detection of any object
Import Libraries: Import the necessary libraries, including torch for PyTorch, models and transforms from torchvision, and cv2 for OpenCV.
Load Pre-trained Model: Load the pre-trained Faster R-CNN model with a ResNet-50 backbone using models.detection.fasterrcnn_resnet50_fpn(pretrained=True). This model is pre-trained on the COCO dataset and is capable of detecting objects in images.
Load and Preprocess Image: Load an image using OpenCV (cv2.imread) and convert it from BGR to RGB format. Then, use F.to_tensor from torchvision.transforms to convert the image to a PyTorch tensor and add a batch dimension using unsqueeze(0).
Perform Inference: Perform inference on the preprocessed image using the loaded model. Use torch.no_grad() to disable gradient calculations, as inference does not require gradient computation. The output contains the predicted bounding boxes, scores, and labels for each detected object.
Visualize Detections: Iterate over the detected objects (assuming one image in batch) and draw bounding boxes around them using cv2.rectangle. Also, display the label and score for each detection using cv2.putText.
Display Result: Display the image with the bounding boxes and text using cv2.imshow, cv2.waitKey, and cv2.destroyAllWindows.
