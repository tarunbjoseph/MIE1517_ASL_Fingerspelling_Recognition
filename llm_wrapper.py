import os
from groq import Groq

def enhance_text_with_llm(raw_text, mode):
    """
    Takes raw fingerspelled characters from the vision model and maps 
    them to full intents using the Groq API.
    """
    # Guard clause to pass through system errors directly
    if "Error:" in raw_text or "Please upload" in raw_text:
        return raw_text
        
    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "System Error: GROQ_API_KEY secret not found in environment."
        
    client = Groq(api_key=api_key)
    
    # Define constraints based on UI mode selection
    if mode == "Emergency/Medical":
        system_prompt = """
        You are a real-time ASL fingerspelling translator in Emergency mode.
        The input is a raw, potentially noisy character prediction from a vision model.
        Map the input strictly to these phrases:
        - If input contains '1', output: 'I need water.'
        - If input contains '2', output: 'I need a doctor.'
        - If input contains 'f' or 'F', output: 'I am finished speaking.'
        - If input contains 'l' or 'L', output: 'Please call my emergency contact.'
        If none of these match, output: 'System Error: Unrecognized Sign.'
        Output ONLY the exact phrase, nothing else.
        """
    else:
        system_prompt = """
        You are an ASL fingerspelling translator.
        The input is raw character predictions from a vision model (e.g., '1', '2', 'f').
        Map '1' to 'Yes', '2' to 'No', 'f' to 'Thank you', and 'l' to 'Hello'.
        Output ONLY the exact translated word.
        """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
            model="llama-3.1-8b-instant", # Extremely fast for low latency inference
            temperature=0.0, # Zero creativity to ensure strict mapping
            max_tokens=20,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM API Error: {str(e)}"
