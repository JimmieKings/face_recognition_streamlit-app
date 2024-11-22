import streamlit as st
import joblib
import numpy as np
from keras_facenet import FaceNet
from PIL import Image
import cv2

# Loading the model and label encoder
try:
    model = joblib.load('/Users/kingoriwangui/fr_streamlit_app/face_recognition_model.pkl')
    label_encoder = joblib.load('/Users/kingoriwangui/fr_streamlit_app/label_encoder.pkl')
    st.write("✅ Model and label encoder loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or label encoder: {e}")

# Loading FaceNet for embedding extraction
try:
    embedder = FaceNet()
    st.write("✅ FaceNet embedding model loaded successfully!")
except Exception as e:
    st.error(f"Error loading FaceNet: {e}")

# Function for face extraction & preprocessing
def extract_face(image, target_size=(160, 160)):
    try:
        # Converting to grayscale
        gray = image.convert('L')
        
        # Loading Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            st.error("❌ Haar Cascade not loaded properly!")
            return None
        
        # Detect face
        faces = face_cascade.detectMultiScale(np.array(gray), scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = image.crop((x, y, x + w, y + h)).resize(target_size)
            return np.array(face)
        else:
            st.warning("No faces detected in the image.")
            return None
    except Exception as e:
        st.error(f"Error during face extraction: {e}")
        return None

# Streamlit app title and description
st.title("🎭 Fun Face Recognition App!")
st.write("Upload your image, and let the app identify the face. Try it out and share your results!")

# Upload image section
uploaded_file = st.file_uploader("📤 Upload an image (JPG or PNG)", type=['jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Processing the image..."):
        face = extract_face(image)
        if face is not None:
            st.write("✅ Face successfully extracted. Generating embedding...")
            try:
                # Generate embedding
                embedding = embedder.embeddings(np.expand_dims(face, axis=0))[0]
                st.write("✅ Embedding generated successfully!")

                # Predict using the trained model
                prediction = model.predict([embedding])
                label = label_encoder.inverse_transform(prediction)
                st.success(f"✅ Identified as: **{label[0]}**")
                st.balloons()
            except Exception as e:
                st.error(f"Error during embedding generation or prediction: {e}")
        else:
            st.error("❌ No face detected in the image. Please try again with a clearer image.")

# Sidebar information
st.sidebar.header("About the App")
st.sidebar.write("This app uses advanced machine learning and FaceNet technology to identify faces. Have fun experimenting with your images!")

st.sidebar.subheader("Tips for better results:")
st.sidebar.write("""
- Use well-lit images.
- Ensure the face is fully visible.
- Avoid blurry or heavily cropped photos.
""")