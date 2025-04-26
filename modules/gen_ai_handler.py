from openai import OpenAI
import base64
from dotenv import load_dotenv
import io


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # You can change the format if necessary (e.g., "JPEG")
    
    # Get the byte data of the image
    image_bytes = buffered.getvalue()

    # Encode the image bytes to base64 and decode to utf-8
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    # Now you can use the base64_string, for example, to send it as part of a response or store it
    return base64_string
    
# Function which call the API for image captioning
def gpt_api(client,system_prompt,img_url, title):
    encoded_image = encode_image(img_url)
    response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
                }
                },
            ],
        },
        {
            "role": "user",
            "content": title
        }
    ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content

def get_image_analysis_v1(image_path):

    load_dotenv()

    client = OpenAI()

    system_prompt = '''
    You will be provided with an image of swan teak wood.
    The image may contain regions of both Sapwood and Heartwood or only one of them.
    There are three grades: A, B, and C, based on the percentage of Sapwood and Heartwood.
    Heartwood percentage above 90% qualifies as Grade A.
    provide the Sapwood and Heartwood percentage.
    provide the Grade of swan teak wood at last in short.
    Additionally, provide the teak wood type.
    '''
    title = "Image of Swan Teak Wood"

    return gpt_api(client,system_prompt,image_path,title)

def get_image_analysis_v2(image):

    load_dotenv()

    client = OpenAI()

    system_prompt = '''
    You will be provided with an image of swan teak wood.
    The image may contain regions of both Sapwood and Heartwood or only one of them.
    There are three grades: A, B, and C, based on the percentage of Sapwood and Heartwood.
    Heartwood percentage above 90% qualifies as Grade A.
    provide the Sapwood and Heartwood percentage.
    provide the Grade of swan teak wood at last in short.
    Answer should be in maximum 40 words.
    '''
    title = "Image of Swan Teak Wood"

    return gpt_api(client,system_prompt,image,title)


