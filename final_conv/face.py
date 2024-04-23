import streamlit as st
import cv2

def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Streamlit App")
    st.caption("Powered by OpenCV, Streamlit")
    
    # Try to initialize the camera
    cap = cv2.VideoCapture(0)  # Use camera index 0 for default camera, or change to the appropriate index
    if not cap.isOpened():
        st.error("Failed to open camera.")
        return
    
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
