import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Draw rectangle
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Region of interest
    roi = frame[y1:y2, x1:x2]

    # Preprocessing
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28,28))
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    img = thresh / 255.0
    img = img.reshape(1,28,28,1)

    # Prediction
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    # Display result
    cv2.putText(
        frame,
        f"Digit: {digit}",
        (100, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Handwritten Digit Recognition", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
