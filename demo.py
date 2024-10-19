import cv2
from PIL import Image
import clip
import torch
import numpy as np
import threading
from src.models.base import EmotionCLIP
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
yolo_model = YOLO("yolo11n.pt")

emotion_classes = ["happy expression", "sad expression", "neutral expression", "angry expression"]

device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_texts = clip.tokenize(emotion_classes).to(device)

emotionclip_model = EmotionCLIP(backbone_checkpoint=None)
ckpt = torch.load("exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt", map_location='cpu')
emotionclip_model.load_state_dict(ckpt['model'], strict=True)
emotionclip_model.eval()

label = ""
combined_mask = np.zeros((480, 640), dtype=np.uint8)


def detect_emotion():
    global combined_mask
    yolo_results = yolo_model(frame)
    person_class_id = 0
    # View results
    if yolo_results:
        for r in yolo_results:
            if r.masks:  # Ensure that there are masks in the result
                masks = r.masks  # Get the masks
                classes = r.boxes.cls  # Get the class predictions for each mask (assuming they are stored in r.boxes.cls)

                # Filter masks that correspond to the "person" class
                person_masks = [mask for mask, cls in zip(masks, classes) if cls == person_class_id]
                for mask in person_masks:
                    print("AAAAAAAAAAAAAAAAAAAAAA")
                    combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
            else:
                # Initialize an empty mask (same height and width as the frame) filled with zeros
                height, width = frame.shape[:2]  # Get the frame dimensions
                combined_mask = np.zeros((height, width), dtype=np.uint8)  # Create an empty mask filled with 0s

    else:
        # Initialize an empty mask (same height and width as the frame) filled with zeros
        height, width = frame.shape[:2]  # Get the frame dimensions
        combined_mask = np.zeros((height, width), dtype=np.uint8)  # Create an empty mask filled with 0s


    frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float()  # Convert to float32 and normalize
    mask_tensor = torch.from_numpy(combined_mask)
    #image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)


    with torch.no_grad():
        image_features = emotionclip_model.encode_image(frame_tensor, mask_tensor)


    image_features /= image_features.norm(dim=-1, keepdim=True)


    with torch.no_grad():
        emotion_features = emotionclip_model.encode_text(emotion_texts)
        emotion_features /= emotion_features.norm(dim=-1, keepdim=True)



    logits_per_image = (100.0 * image_features @ emotion_features.T).softmax(dim=-1)     # size: (B, class_num)

    probs = logits_per_image.cpu().numpy()[0]
    for i, emotion in enumerate(emotion_classes):
        print(f"Probability of {emotion}: {probs[i]*100:.2f}%")
    # Using global because of multithreading
    global label
    max_prob_idx = np.argmax(probs)
    max_emotion = emotion_classes[max_prob_idx]
    max_prob = f"{probs[max_prob_idx]*100:.2f}%"
    print(f"Prediction: {max_emotion}\n")
    label = f"{max_emotion}, {max_prob}"

def draw_label():
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # White color
    thickness = 2
    label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

    # Set the position of the label
    label_x = 0
    label_y = 20

    # Draw the label background (optional, makes text more readable)
    cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5), (label_x + label_size[0], label_y + 5), (0, 255, 0),
                  cv2.FILLED)

    # Put the label on top of the rectangle
    cv2.putText(frame, label, (label_x, label_y), font, font_scale, font_color, thickness)

# opening webcam
cap = cv2.VideoCapture(0)
frame_count = 0  # Initialize a frame counter
processing_thread = None  # To keep track of the current processing thread
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Only run every 10 frames
    if frame_count % 1 == 0:
        # If there is no ongoing thread, start a new one to process the frame
        if processing_thread is None or not processing_thread.is_alive():
            processing_thread = threading.Thread(target=detect_emotion, args=())
            processing_thread.start()

    draw_label()
    cv2.imshow('webcam', frame)
    cv2.imshow('mask', combined_mask)

    frame_count += 1  # Increment the frame counter


cap.release()
cv2.destroyAllWindows()

