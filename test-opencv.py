import cv2
import numpy as np
from src.models.base import EmotionCLIP
import torch
import joblib

CLASS_NAMES = [
    'peace',
    'affection',
    'esteem',
    'anticipation',
    'engagement',
    'confidence',
    'happiness',
    'pleasure',
    'excitement',
    'surprise',
    'sympathy',
    'doubt confusion',
    'disconnection',
    'fatigue',
    'embarrassment',
    'yearning',
    'disapproval',
    'aversion',
    'annoyance',
    'anger',
    'sensitivity',
    'sadness',
    'disquietment',
    'fear',
    'pain',
    'suffering'
]


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)

# Load the pre-trained model
model = EmotionCLIP(backbone_checkpoint=None)
ckpt = torch.load("C:/Users/Krist/OneDrive/Documents/GitHub/EmotionCLIP-Assignment/exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt", map_location='cpu')
model.load_state_dict(ckpt['model'], strict=True)
model.eval()

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    mask = np.zeros_like(gray_image)  # Create a blank mask (same size as the frame)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)  # Draw bounding box
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255), -1)  # Fill mask with white on detected faces    

    return faces, mask

def draw_labels(label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)  # White color
    thickness = 2
    label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    
    label_lines = label.split('\n')
    label_y = 10
    label_x = 10

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)  # White color
    thickness = 2
    line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 5  # Approximate line height with some padding
    
     # Draw each line of the label
    for i, line in enumerate(label_lines):
        # Compute the position for the current line
        current_y = label_y + i * line_height

        # Draw background rectangle for each line (optional)
        label_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        cv2.rectangle(video_frame, (label_x, current_y - label_size[1]), (label_x + label_size[0], current_y), (0, 255, 0), cv2.FILLED)
        
        # Put the text on the image
        cv2.putText(video_frame, line, (label_x, current_y), font, font_scale, font_color, thickness)

predictions_counter = 1
while True:
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    if predictions_counter == 1:
            
        # shape of visual mask and frames in linear_eval : (128, 8, 3, 224, 224)

        faces, face_mask = detect_bounding_box(video_frame)  # detect faces and generate mask
        

        # Resize frame and mask to match the input size expected by the model
        frame = cv2.resize(video_frame, (224, 224))
        mask = cv2.resize(face_mask, (224, 224))

        frame = frame.transpose(2, 0, 1)
        
        # Convert to torch tensors
        frame_tensor = torch.from_numpy(frame).float()  # Convert to float32 and normalize
        mask_tensor = torch.from_numpy(mask)  # Convert to float32 and normalize
        print(frame_tensor)
        print(mask_tensor)
        
        # Transpose frame and mask size to (3, 224, 224) and (1, 224, 224)
        #mask_tensor = mask_tensor.unsqueeze(0)
        #print(mask_tensor.shape)

        # # Add batch dimension: (1, 3, 224, 224)
        # video_frame_tensor = video_frame_tensor.unsqueeze(0)
        # face_mask_tensor = face_mask_tensor.unsqueeze(0)

        #frame_tensor = frame_tensor.repeat(1, 8, 1, 1, 1)
        #mask_tensor = mask_tensor.repeat(1, 8, 1, 1, 1)
        #print(frame_tensor.shape)
        #print(mask_tensor.shape)

        C, H, W = frame.shape
        frame_tensor = frame_tensor.reshape(-1, C, H, W)
        mask_tensor = mask_tensor.reshape(-1, H, W)
        features = model.encode_image(frame_tensor, mask_tensor)
        features = features.detach().numpy()
        print(f"features shape: {features.shape}")
        #with torch.no_grad():
        #    output = model(features)

        linear_clf = joblib.load('linear_clf_model.joblib')
        predictions = linear_clf.predict(features)
        print(f"Prediction: {predictions}")

        counter = 0
        label = ''
        for pred in predictions[0]:
            if pred == 1:
                label += str(CLASS_NAMES[counter]) + '\n'
            counter += 1
        
        predictions_counter = 0

    draw_labels(label)
    predictions_counter += 1

        
    cv2.imshow("Face Detection", video_frame)  # display the frame with bounding boxes
    cv2.imshow("Face Mask", face_mask)  # display the mask of the faces

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
