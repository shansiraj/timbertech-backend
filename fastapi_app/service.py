from PIL import Image
from modules.image_handler import resize_image_maintain_aspect_ratio, divide_image_into_patches
from modules.model_handler import load_model
from modules.image_handler import load_image, convert_to_displayable_image
from modules.preprocessor import preprocessing
from modules.teak_grader import get_grade_results, create_output_prob_json, get_accumulated_prob
from modules.gen_ai_handler import get_image_analysis_v2
from modules.teak_pricer import get_price,get_currency


# Grade to numeric map
grade_to_num = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
num_to_grade = {v: k for k, v in grade_to_num.items()}

# Simulated model output (replace this with real image classification logic)
def classify_wood_image(image: Image.Image):

    grades = []
    probabilities = []
    prices = []

    selected_model = 'ResNet18'
    model = load_model(selected_model)
   
    # Divide the resized image into 224x224 patches
    patches = divide_image_into_patches(image, 224)
    print ("Number of patches: " + str(len(patches)))

    for patch in patches:
        # Step 1: Preprocess each patch
        preprocessed_image = preprocessing(patch)

        # Step 2: Predict grade and probability
        grade, grade_text, probabilities_np = get_grade_results(model, preprocessed_image)

        # Step 3: Convert grade to number for averaging later
        numeric_grade = grade_to_num[grade_text]

        # Step 4: Calculate price
        price = get_price(grade_text)

        # Step 5: Accumulate results
        grades.append(numeric_grade)
        probabilities.append(probabilities_np)
        prices.append(price)

    # Step 6: Aggregate results
    average_numeric_grade = round(sum(grades) / len(grades))
    final_grade_text = num_to_grade.get(average_numeric_grade, 'D')  # fallback to D

    total_price = sum(prices)

    accumulated_probability = get_accumulated_prob(probabilities)

    # Create a DataFrame for grades and probabilities
    grade_probability_json = create_output_prob_json(accumulated_probability)

    description_text = get_image_analysis_v2(image)

    # Fake classification logic - Replace with actual model prediction logic
    final_grade = final_grade_text
    price = str(total_price) + " " + get_currency()
    grade_probability = grade_probability_json
    description = description_text
    
    # Simulating model prediction
    return final_grade, price, grade_probability, description