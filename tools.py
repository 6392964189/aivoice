import cv2
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def capture_image() -> str:
    """
    Captures one frame from the default webcam, resizes it,
    encodes it as Base64 JPEG (raw string) and returns it.
    Works on Windows using CAP_DSHOW backend.
    """
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Use DirectShow for Windows
        if cap.isOpened():
            for _ in range(10):  # Warm up camera
                cap.read()
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                continue

            # Optional: Save captured image for debug
            cv2.imwrite("sample.jpg", frame)

            # Encode image to base64
            ret, buf = cv2.imencode('.jpg', frame)
            if ret:
                img_b64 = base64.b64encode(buf).decode('utf-8')
                return img_b64
    raise RuntimeError("Could not open any webcam (tried indices 0-3)")


def analyze_image_with_query(query: str) -> str:
    """
    Captures image and sends the query + image to Groq's Vision API.
    Returns the analysis result from the model.
    """
    # Capture image as base64 string
    img_b64 = capture_image()

    # Set model
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"

    # Validate input
    if not query or not img_b64:
        return "Error: both 'query' and image are required."

    # Create Groq client
    client = Groq()

    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }},
            ],
        }
    ]

    # Get response from Groq
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    # Return only the content
    return chat_completion.choices[0].message.content


# Sample run
#if __name__ == "__main__":
    #query = "How many people do you see?"
    #try:
        #result = analyze_image_with_query(query)
       # print("Result from Groq Vision Model:\n", result)
    #except Exception as e:
        #print("Error:", str(e))
