import os
import io
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image

load_dotenv()

def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.title("Gemini Model")
st.write("")
gemini_pro_or_gemini_pro_vision = st.radio("Select Model", ["Gemini Pro", "Gemini Pro Vision"])

def main():
    try:
        if gemini_pro_or_gemini_pro_vision == "Gemini Pro":
            with st.expander("Gemini Pro"):
                st.header("Interact with Gemini Pro")
                st.write("")

                message = st.text_input("Enter message: ", placeholder="Message", label_visibility='visible')
                model = genai.GenerativeModel('gemini-pro')

                if st.button("SEND", use_container_width=True):
                    response = model.generate_content(message)
                    st.write("")
                    st.subheader(":blue[Response]")
                    st.write("")
                    st.markdown(response.text)

        else:
            with st.expander("Gemini Pro Vision"):
                st.header("Interact with Gemini Pro Vision")
                st.write("")

            image_prompt = st.text_input("Interact with the Image", placeholder="Message", label_visibility="visible")
            uploaded_file = st.file_uploader("Choose and Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

            if uploaded_file is not None:
                st.image(Image.open(uploaded_file), use_column_width=True)

                st.markdown("""
                    <style>
                            img {
                                border-radius: 10px;
                            }
                    </style>
                    """, unsafe_allow_html=True)

            if st.button("GET RESPONSE", use_container_width=True):
                model = genai.GenerativeModel("gemini-pro-vision")

                if uploaded_file is not None:
                    if image_prompt != "":
                        image = Image.open(uploaded_file)

                        response = model.generate_content(
                            glm.Content(
                                parts=[
                                    glm.Part(text=image_prompt),
                                    glm.Part(
                                        inline_data=glm.Blob(
                                            mime_type="image/jpeg",
                                            data=image_to_byte_array(image)
                                        )
                                    )
                                ]
                            )
                        )

                        response.resolve()

                        st.write("")
                        st.write(":blue[Response]")
                        st.write("")

                        st.markdown(response.text)

                    else:
                        st.write("")
                        st.subheader(":red[Please Provide a prompt]")

                else:
                    st.write("")
                    st.subheader(":red[Please Provide an image]")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
