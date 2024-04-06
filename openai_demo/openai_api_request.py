"""
This script is designed to mimic the OpenAI API interface with CogVLM & CogAgent Chat
It demonstrates how to integrate image and text-based input to generate a response.
Currently, the model can only handle a single image.
Therefore, do not use this script to process multiple images in one conversation. (includes images from history)
And it only works on the chat model, not the base model.
"""
import requests
import json
import jsonlines
import base64
from tqdm import tqdm


base_url = "http://127.0.0.1:8000"


def create_chat_completion(model, messages, temperature=0.8, max_tokens=2048, top_p=0.8, use_stream=False):
    """
    This function sends a request to the chat API to generate a response based on the given messages.

    Args:
        model (str): The name of the model to use for generating the response.
        messages (list): A list of message dictionaries representing the conversation history.
        temperature (float): Controls randomness in response generation. Higher values lead to more random responses.
        max_tokens (int): The maximum length of the generated response.
        top_p (float): Controls diversity of response by filtering less likely options.
        use_stream (bool): Determines whether to use a streaming response or a single response.

    The function constructs a JSON payload with the specified parameters and sends a POST request to the API.
    It then handles the response, either as a stream (for ongoing responses) or a single message.
    """

    data = {
        "model": model,
        "messages": messages,
        "stream": use_stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    if response.status_code == 200:
        parsed_content = []
        if use_stream:
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        parsed_content.append(content)
                    except:
                        print("Special Token:", decoded_line)
            if not parsed_content:
                print("Streaming failed")
                return None
            return "".join(parsed_content)
        else:
            # 处理非流式响应
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            print(content)
            return content
    else:
        print("Error:", response.status_code)
        return None


def encode_image(image_path):
    """
    Encodes an image file into a base64 string.
    Args:
        image_path (str): The path to the image file.

    This function opens the specified image file, reads its content, and encodes it into a base64 string.
    The base64 encoding is used to send images over HTTP as text.
    """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def simple_captioning(use_stream=True, img_path=None, tags_string:str=None, prompt_template:str=None, temperature=0.8, max_tokens=2048, top_p=0.8) -> str:
    """
    Facilitates a simple chat interaction involving an image.
    Args:
        use_stream (bool): Specifies whether to use streaming for chat responses.
        img_path (str): Path to the image file to be included in the chat.
        tags_string (str): A string of tags to be included in the chat.
        
        prompt_template(str): A string of prompt template to be included in the chat. The prompt should include $tags placeholder.
        If not specified, "Please describe the image in very simple caption, using given tags: $tags" will be used.
    """
    
    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    if prompt_template is None:
        prompt_template = "Please describe the image in very simple caption, using given tags: $tags. Describe situation or action, image's abnormal features with given tags. You must create short sentences, not by listing tags."
    prompt = prompt_template.replace("$tags", str(tags_string).replace("_", " "))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    },
                },
            ],
        },
    ]
    return create_chat_completion("cogvlm-chat-17b", messages=messages, use_stream=use_stream, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

def bulk_captioning(image_paths_json:str = None, tags_json:str = None, save_path:str=None, prompt_template:str=None, temperature=0.8, max_tokens=2048, top_p=0.8):
    """
    Facilitates a simple chat interaction involving multiple images.
    Args:
        image_paths_json (str): A string of json list of image paths to be included in the chat.
        tags_json (str): A string of json list of tags to be included in the chat.
        
        prompt_template(str): A string of prompt template to be included in the chat. The prompt should include $tags placeholder.
        If not specified, "Please describe the image in very simple caption, using given tags: $tags, but do not directly list all given tags. Try to describe situation or action within given tags." will be used.
    """
    with open(image_paths_json, 'r', encoding='utf-8') as f:
        image_paths = json.load(f)
    with open(tags_json, 'r', encoding='utf-8') as f:
        tags = json.load(f)
    processed_images = set()
    # Check if save_path exists and load already processed images
    try:
        with jsonlines.open(save_path, mode='r') as reader:
            for item in reader:
                processed_images.add(item['image_path'])
    except FileNotFoundError:
        pass  # If file doesn't exist, we start afresh
    print("Loaded images and tags")
    with jsonlines.open(save_path, mode='a') as writer:
        for i in tqdm(range(len(image_paths))):
            if image_paths[i] in processed_images:
                print(f"Skipping already processed image: {image_paths[i]}")
                continue  # Skip already processed images
            tag_string = tags[i]
            tag_string = [t for t in tag_string if len(t) > 1]
            print(tag_string)
            result = simple_captioning(use_stream=False, img_path=image_paths[i], tags_string=tag_string, prompt_template=prompt_template, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
            if not result:
                print("result is none")
                raise RuntimeError("Failed")
            print("result", result)
            writer.write({'image_path': image_paths[i], 'result': result})
            writer._fp.flush()
    print(f"Wrote down results in {save_path}")

def simple_image_chat(use_stream=True, img_path=None):
    """
    Facilitates a simple chat interaction involving an image.

    Args:
        use_stream (bool): Specifies whether to use streaming for chat responses.
        img_path (str): Path to the image file to be included in the chat.

    This function encodes the specified image and constructs a predefined conversation involving the image.
    It then calls `create_chat_completion` to generate a response from the model.
    The conversation includes asking about the content of the image and a follow-up question.
    """

    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": "What’s in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": "The image displays a wooden boardwalk extending through a vibrant green grassy wetland. The sky is partly cloudy with soft, wispy clouds, indicating nice weather. Vegetation is seen on either side of the boardwalk, and trees are present in the background, suggesting that this area might be a natural reserve or park designed for ecological preservation and outdoor recreation. The boardwalk allows visitors to explore the area without disturbing the natural habitat.",
        },
        {
            "role": "user",
            "content": "Do you think this is a spring or winter photo?"
        },
    ]
    create_chat_completion("cogvlm-chat-17b", messages=messages, use_stream=use_stream)


if __name__ == "__main__":
    # simple_image_chat(use_stream=False, img_path="demo.jpg")
    import argparse
    parser = argparse.ArgumentParser()
    # bulk captioning
    parser.add_argument("--bulk_captioning", action="store_true", help="Enable bulk captioning mode")
    parser.add_argument("--image_paths_json", type=str, help="A string of json list of image paths to be included in the chat")
    parser.add_argument("--tags_json", type=str, help="A string of json list of tags to be included in the chat")
    parser.add_argument("--save_path", type=str, help="Path to save the results")
    # simple captioning
    parser.add_argument("--simple_captioning", action="store_true", help="Enable simple captioning mode")
    parser.add_argument("--img_path", type=str, help="Path to the image file to be included in the chat")
    parser.add_argument("--tags_string", type=str, help="A string of tags to be included in the chat")
    parser.add_argument("--prompt_template", type=str, help="A string of prompt template to be included in the chat", default=None)
    # common
    parser.add_argument("--use_stream", action="store_true", help="Specifies whether to use streaming for chat responses")
    parser.add_argument("--temperature", type=float, default=0.8, help="Controls randomness in response generation. Higher values lead to more random responses.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="The maximum length of the generated response.")
    parser.add_argument("--top_p", type=float, default=0.8, help="Controls diversity of response by filtering less likely options.")
    parser.add_argument("--port", type=int, default=8000, help="Local port to use.")
    args = parser.parse_args()
    # port
    if args.port:
        base_url = f"http://127.0.0.1:{args.port}"
    if args.bulk_captioning:
        bulk_captioning(image_paths_json=args.image_paths_json, tags_json=args.tags_json, save_path=args.save_path, prompt_template=args.prompt_template, temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p)
    elif args.simple_captioning:
        simple_captioning(use_stream=args.use_stream, img_path=args.img_path, tags_string=args.tags_string, prompt_template=args.prompt_template, temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p)
    else:
        simple_image_chat(use_stream=args.use_stream, img_path=args.img_path)
