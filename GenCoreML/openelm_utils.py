import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from typing import Optional 

def load_openelm_model(
    model_name: str = "apple/OpenELM-270M",
    device: Optional[str] = None,
) -> tuple:
    """
    Load OpenELM model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cpu', 'cuda', 'mps', or None for auto)
        hf_access_token: HuggingFace access token if needed
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"🔄 Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    return model, tokenizer, device

def generate_text(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    max_length: int = 256,
    max_new_tokens: int = None,
    temperature: float = 0.9,
    top_p: float = 0.9,
    do_sample: bool = True,
    num_beams: int = 1,
    repetition_penalty: float = 1.2,
    pad_token_id: Optional[int] = None
) -> dict:
    """
    Generate text using OpenELM model.
    
    Args:
        prompt: Input text prompt
        model: Loaded OpenELM model
        tokenizer: Loaded tokenizer
        device: Device model is on
        max_length: Maximum total sequence length
        max_new_tokens: Maximum new tokens to generate (overrides max_length)
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to use sampling
        num_beams: Number of beams for beam search
        repetition_penalty: Repetition penalty factor
        pad_token_id: Padding token ID
        
    Returns:
        Dictionary with generated text, timing info, and metadata
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    
    print(f"🧪 Generating text for prompt: '{prompt}'")
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    
    # Set max_new_tokens if not provided
    if max_new_tokens is None:
        max_new_tokens = max_length - input_length
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    tokens_generated = outputs.shape[1] - input_length
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "new_text": new_text,
        "tokens_generated": tokens_generated,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
        "input_length": input_length,
        "output_length": outputs.shape[1]
    }

def quick_generate(
    prompt: str,
    model_name: str = "apple/OpenELM-270M",
    max_new_tokens: int = 50,
    temperature: float = 0.9,
    device: Optional[str] = None,
    hf_access_token: Optional[str] = None
) -> str:
    """
    Quick text generation with minimal setup.
    
    Args:
        prompt: Input text prompt
        model_name: HuggingFace model identifier
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        device: Device to use
        hf_access_token: HuggingFace access token
        
    Returns:
        Generated text string
    """
    model, tokenizer, device = load_openelm_model(
        model_name=model_name,
        device=device,
        hf_access_token=hf_access_token
    )
    
    result = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    print(f"⚡ Generated {result['tokens_generated']} tokens in {result['generation_time']:.2f}s")
    print(f"🚀 Speed: {result['tokens_per_second']:.1f} tokens/second")
    print(f"📝 Output: {result['new_text']}")
    
    return result['generated_text']

def batch_generate(
    prompts: list,
    model_name: str = "apple/OpenELM-270M",
    max_new_tokens: int = 50,
    temperature: float = 0.9,
    device: Optional[str] = None,
    hf_access_token: Optional[str] = None
) -> list:
    """
    Generate text for multiple prompts efficiently.
    
    Args:
        prompts: List of input prompts
        model_name: HuggingFace model identifier
        max_new_tokens: Maximum new tokens to generate per prompt
        temperature: Sampling temperature
        device: Device to use
        hf_access_token: HuggingFace access token
        
    Returns:
        List of generation results
    """
    model, tokenizer, device = load_openelm_model(
        model_name=model_name,
        device=device,
        hf_access_token=hf_access_token
    )
    
    results = []
    total_start = time.time()
    
    for i, prompt in enumerate(prompts):
        print(f"\n📋 Processing prompt {i+1}/{len(prompts)}")
        result = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        results.append(result)
        print(f"✅ Completed in {result['generation_time']:.2f}s")
    
    total_time = time.time() - total_start
    print(f"\n🎯 Batch completed in {total_time:.2f}s")
    
    return results

def interactive_chat(
    model_name: str = "apple/OpenELM-270M",
    max_new_tokens: int = 100,
    temperature: float = 0.9,
    device: Optional[str] = None,
    hf_access_token: Optional[str] = None
):
    """
    Interactive chat session with OpenELM.
    
    Args:
        model_name: HuggingFace model identifier
        max_new_tokens: Maximum new tokens per response
        temperature: Sampling temperature
        device: Device to use
        hf_access_token: HuggingFace access token
    """
    model, tokenizer, device = load_openelm_model(
        model_name=model_name,
        device=device,
        hf_access_token=hf_access_token
    )
    
    print("💬 Interactive OpenELM Chat (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        prompt = input("\n👤 You: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not prompt:
            continue
        
        result = generate_text(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        print(f"🤖 OpenELM: {result['new_text']}")
        print(f"   ({result['tokens_generated']} tokens, {result['tokens_per_second']:.1f} tok/s)")
