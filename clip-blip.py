import cv2
#from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import clip
import torch
import numpy as np
import threading

# CLIP for detecting emotion (for giving the prob of each class)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

emotion_classes = ["happy expression", "sad expression", "neutral expression", "angry expression"]
emotion_texts = clip.tokenize(emotion_classes).to(device)

# BLIP for generating caption per frame
#processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


label = ""

#def generate_caption():
    #img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    #inputs = processor(images=img, return_tensors="pt")
    #out = model.generate(**inputs)
    #caption = processor.decode(out[0], skip_special_tokens=True)
    #print(caption)

    #return caption


def detect_emotion():
    image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    #caption_input = clip.tokenize([caption]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        #caption_features = clip_model.encode_text(caption_input)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    #caption_features /= caption_features.norm(dim=-1, keepdim=True)

    #combined_features = (image_features + caption_features) / 2

    with torch.no_grad():
        emotion_features = clip_model.encode_text(emotion_texts)
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



def process_frame():
    #caption = generate_caption()
    detect_emotion()


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
    if frame_count % 10 == 0:
        # If there is no ongoing thread, start a new one to process the frame
        if processing_thread is None or not processing_thread.is_alive():
            processing_thread = threading.Thread(target=process_frame, args=())
            processing_thread.start()

    draw_label()
    cv2.imshow('webcam', frame)
    frame_count += 1  # Increment the frame counter


cap.release()
cv2.destroyAllWindows()