import cv2
import mediapipe as mp
import math
import pyautogui

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to map a value from one range to another
def map_value(value, left_min, left_max, right_min, right_max):
    left_span = left_max - left_min
    right_span = right_max - right_min
                                            
    # Convert the left range into a 0-1 range (float)
    value_scaled = float(value - left_min) / float(left_span)

    # Convert the 0-1 range into a value in the right range.
    return right_min + (value_scaled * right_span)

# Main function
def main():
    cap = cv2.VideoCapture(0)
    volume_increment = 3  # Change this value to adjust the volume increment

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_x, thumb_y = int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])
                index_x, index_y = int(index.x * frame.shape[1]), int(index.y * frame.shape[0])

                # Calculate distance between thumb and index finger
                distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)

                # Check if the distance is small (pinching gesture)
                if distance < 50:
                    pyautogui.press('volumedown', presses=volume_increment)
                    
                # Check if the distance is large (spreading gesture)
                elif distance > 100:
                    pyautogui.press('volumeup', presses=volume_increment)

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw line between thumb and index finger
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 0, 255), thickness=2)
                
        # Resize the frame to adjust the video size according to your preference
        new_width = 840  # Specify the desired width
        new_height = 580  # Specify the desired height
        frame = cv2.resize(frame, (new_width, new_height))

        cv2.imshow('Gesture Volume Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
