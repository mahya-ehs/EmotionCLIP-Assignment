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

emotion_classes = ["happy", "sad", "neutral", "angry"]

device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_texts = clip.tokenize(emotion_classes).to(device)

emotionclip_model = EmotionCLIP(backbone_checkpoint=None)
ckpt = torch.load("exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt", map_location='cpu')
emotionclip_model.load_state_dict(ckpt['model'], strict=True)
emotionclip_model.eval()

height = 224
width = 224
label = ""
combined_mask = torch.zeros((height, width), dtype=torch.uint8)
person_boxes = []


def bbox_to_mask(bbox: list[float], target_shape: tuple[int, int]) -> torch.Tensor:
    mask = torch.zeros(target_shape[1], target_shape[0])
    if len(bbox) == 0:
        return mask
    mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1
    print(f"mask {mask.shape}")
    return mask

def detect_emotion():
    global combined_mask
    global person_boxes
    height, width = frame.shape[:2]
    combined_mask = torch.zeros((height, width), dtype=torch.uint8)

    yolo_results = yolo_model(frame)
    person_class_id = 0
    # View results
    if yolo_results:
        for r in yolo_results:
            if r.boxes:  # Ensure that there are masks in the result
                boxes = r.boxes.xyxy  # Get the boxes
                classes = r.boxes.cls  # Get the class predictions for each mask (assuming they are stored in r.boxes.cls)

                # Filter masks that correspond to the "person" class
                person_boxes = []
                for box, cls in zip(boxes, classes):
                    if cls == person_class_id:
                        person_boxes.append(box.tolist())

                person_masks = []
                for box in person_boxes:
                    person_masks.append(bbox_to_mask(box, (width, height))) # The given function has height and width swapped for reasons I don't understand

                for mask in person_masks:
                    combined_mask = torch.logical_or(combined_mask, mask).to(torch.uint8)
            else:
                # Initialize an empty mask (same height and width as the frame) filled with zeros
                combined_mask = torch.zeros((height, width), dtype=torch.uint8)  # Create an empty mask filled with 0s
    else:
        # Initialize an empty mask (same height and width as the frame) filled with zeros
        combined_mask = torch.zeros((height, width), dtype=torch.uint8)  # Create an empty mask filled with 0s


    frame_inner = cv2.resize(frame, (224, 224))
    combined_mask_numpy = cv2.resize(combined_mask.numpy(), (224, 224))
    frame_tensor = torch.from_numpy(frame_inner.transpose(2, 0, 1)).float()  # Convert to float32 and normalize
    mask_tensor = torch.from_numpy(combined_mask_numpy)
    #image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

    C, H, W = frame_inner.transpose(2, 0, 1).shape
    mask_tensor = mask_tensor.unsqueeze(0)
    frame_tensor = frame_tensor.reshape(-1, C, H, W)
    mask_tensor = mask_tensor.reshape(-1, H, W)

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


def draw_bboxes(person_boxes):
    for box in person_boxes:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)


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

    # Only run every 20 frames
    if frame_count % 20 == 0:
        # If there is no ongoing thread, start a new one to process the frame
        if processing_thread is None or not processing_thread.is_alive():
            processing_thread = threading.Thread(target=detect_emotion, args=())
            processing_thread.start()

    draw_label()
    draw_bboxes(person_boxes)
    cv2.imshow('webcam', frame)
    cv2.imshow('mask', combined_mask.numpy())

    frame_count += 1  # Increment the frame counter


cap.release()
cv2.destroyAllWindows()

