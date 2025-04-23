import streamlit as st
from modules.model_handler import load_model
from modules.image_handler import load_image, convert_to_displayable_image
from modules.preprocessor import preprocessing
from modules.model_explainer import explian_image
from modules.teak_grader import get_grade_results, create_output_prob_df
from modules.gen_ai_handler import get_image_analysis_v1
import pandas as pd

def main():

    # Set the page layout to wide
    st.set_page_config(page_title="TimberTech", layout="wide")

    # Sidebar
    with st.sidebar:
        
        st.markdown(
            """
            <style>
            section[data-testid="stSidebar"] h2 {
                color: white !important;
            }
            section[data-testid="stSidebar"] label {
                color: white !important;
                font-weight: bold;
            }
            .stApp {
                background-color: #E7E8E3; /* Light Blue */
            }
            [data-testid="stSidebar"] {
                background-color: #447055; /* Light gray background color */
                padding-top: 0px;
            }
            .center-text {
                text-align: center;
                color: white;
                font-size: 14px;
            }
            .white-line {
                border: 1px solid white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        meta_text = """
                TimberTech is an AI-driven Sawn Teak Wood Grading System designed for the Sri Lankan wood industry. Using CNN-based deep learning, it accurately analyzes and grades sawn teak wood images, ensuring efficient and reliable quality assessment. It also optionally aids in pricing teak wood surfaces based on their grade.
        """
                
        st.sidebar.image("img/logo.png", clamp=True)
        st.markdown(f'<p class="center-text">{meta_text}</p>', unsafe_allow_html=True)
        st.markdown('<hr class="white-line">', unsafe_allow_html=True)
        
    st.title('Sawn Teak Wood Grading System')
    
    st.sidebar.header('User Input')
    uploaded_file = st.sidebar.file_uploader(label="Upload Image:",type=["png","jpg","jpeg"])

    selected_model = 'ResNet18'
    model = load_model(selected_model)

    # Button to trigger the activity
    if st.sidebar.button('Start Analysis',use_container_width=True):

        if uploaded_file:

            original_image = load_image(uploaded_file)

            img_path = f"tmp/temp_image.jpg"  # Temporary path to save image
            original_image.save(img_path)

            preprocessed_image = preprocessing(original_image)

            transformed_image = convert_to_displayable_image(preprocessed_image)


            grade, grade_text, probabilities_np = get_grade_results(model,preprocessed_image)

            # Create interpreted image
            overlayed_image = explian_image(selected_model,model,preprocessed_image ,img_path)

            # Create a DataFrame for grades and probabilities
            styled_df = create_output_prob_df(probabilities_np)

            probabilities_reversed = probabilities_np[::-1]
            

            st.markdown(f"<h3><strong>Identified Grade : {grade_text}</strong></h3>", unsafe_allow_html=True)

            # Display components
            col1, col2,col3,col4 = st.columns([1,1,1,2])

            with col1:
                st.write("Original Image")
                st.image(original_image, use_column_width=False)

            with col2:
                st.write("Transformed Image")
                st.image(transformed_image, use_column_width=False)

            with col3:
                st.write("Interpreted Image")
                st.image(overlayed_image, caption="Grad-CAM Interpretation", use_column_width=False)
            with col4:
                # Display the probability values for each grade
                st.write("Output probabilities")
                st.write(styled_df)
           
            col1, col2= st.columns([3,2])

            with col2:
                # Display output probabilities in a barchart
                st.write("Output probability chart")
                class_labels = ['A', 'B', 'C', 'D']

                # Create a DataFrame with custom labels
                df = pd.DataFrame({
                    'Grade': class_labels,
                    'Probability': probabilities_reversed
                })
                st.bar_chart(df.set_index('Grade')['Probability'])

            with col1:
                # Display the Gen AI analysis
                st.write("Gen AI Textual Analysis")
                with st.spinner('Please wait... Generating Gen AI textual analysis'):
                    # image_analysis = get_image_analysis(img_path)
                    image_analysis = get_image_analysis_v1(original_image)
                st.write(image_analysis)

            

        else:
            st.error("Image is not loaded")

    with st.sidebar:
        st.markdown('<hr class="white-line">', unsafe_allow_html=True)
        st.markdown('<p class="center-text">Developed by D. Shan Siraj</p>', unsafe_allow_html=True)
        st.markdown('<p class="center-text">STNO: COMSCDS231P-023</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

