from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify the Hugging Face model repository
model_path = "girishnP/tamilllamacustom"  # Use the Hugging Face model name

# Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# Function to generate structured story
def generate_structured_story(prompt, max_length=400, temperature=0.7, top_k=50, top_p=0.9):
    """
    Generate a structured Tamil story based on the given prompt.
    
    Args:
        prompt (str): Input text to guide the generation.
        max_length (int): Maximum length of the generated text.
        temperature (float): Controls randomness (higher = more random).
        top_k (int): Limits sampling to the top-k tokens.
        top_p (float): Nucleus sampling (controls diversity).
    
    Returns:
        str: Generated structured story in Tamil.
    """
    # Add structured story instructions to the prompt
    structured_prompt = (
        f"தமிழில் ஒரு சிறுகதை எழுதுங்கள். கதைக்கு ஒரு அறிமுகம், முரண்பாடு, தீர்வு, மற்றும் முடிவு இருக்க வேண்டும். "
        f"கதையின் தலைப்பு: {prompt}"
    )

    # Tokenize the input prompt
    input_ids = tokenizer(structured_prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        repetition_penalty=1.2,  # Penalize repetition
        no_repeat_ngram_size=3   # Avoid repeating 3-grams
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    generated_text = generated_text.replace(structured_prompt, "").strip()
    
    return generated_text

# Example usage
if __name__ == "__main__":
    # Input prompt
    story_prompt = "ஒரு இளைஞன் ஒரு மாய உலகத்தை கண்டுபிடிக்கிறான்"
 
    # Generate a structured Tamil story
    try:
        story = generate_structured_story(story_prompt, max_length=400)
        print("\nGenerated Structured Story:")
        print(story)
    except Exception as e:
        print(f"Error generating text: {e}")
