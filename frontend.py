import streamlit as st
import requests
import base64

# ✅ Set Page Config
st.set_page_config(page_title="Stance Detection - MTAAN", page_icon="🧠", layout="centered")

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# ✅ Set Background Image
set_background("C:/Users/ganta.greshma/stance-detection-app/image.jpg")  # Adjust path as needed

# 🎯 Project Title
st.markdown(
    "<h1 style='text-align: center; color: red;'>Stance Detection - Multi-Target Adversarial Attention Network</h1>",
    unsafe_allow_html=True,
)

# 🎯 Define Target Topics
targets = [
    "Climate Change", "Gun Control", "Feminism", "Abortion", "Healthcare",
    "Immigration", "Animal Rights", "LGBTQ+ Rights", "Cryptocurrency",
    "AI in Society", "Space Exploration", "Other"
]

# ✍ User Input Fields
st.markdown("### Enter Your Text and Select a Target Topic:")
target = st.selectbox("Select Target:", targets, index=0)

# Allow custom target if "Other" is selected
if target == "Other":
    target = st.text_input("Enter your custom target:")

text = st.text_area("Enter Text:", placeholder="Type a statement here...", height=150)

# 🔥 API Backend Connection
BACKEND_URL = "http://127.0.0.1:5000/predict"  # Update this with the actual backend endpoint

def get_prediction(text, target):
    """ Sends text and target to the backend and fetches stance prediction """
    try:
        response = requests.post(BACKEND_URL, json={"text": text, "target": target})
        if response.status_code == 200:
            return response.json().get("stance", "Error: No stance returned")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to backend: {e}"

# 🔥 Predict and Display Output
if st.button("Predict Stance", help="Click to analyze the stance of the text"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text for prediction.")
    else:
        stance = get_prediction(text, target)
        st.success(f"**Prediction:** {stance}")
