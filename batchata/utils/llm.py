def token_count_simple(text):
    """
    An extremely simple, conservative token estimator.
    This uses basic character and word counting with generous overestimation.
    Ideal for quick estimates when accuracy is less important than avoiding underestimation.
    
    Args:
        text (str): The input text to estimate tokens for
        
    Returns:
        int: Conservative estimated token count
    """
    if not text:
        return 0
        
    # Count characters in different Unicode ranges
    latin_chars = sum(1 for c in text if ord(c) < 0x0300)
    cjk_chars = sum(1 for c in text if 0x3000 <= ord(c) <= 0x9FFF)
    emoji_chars = sum(1 for c in text if 0x1F000 <= ord(c) <= 0x1FFFF)
    other_chars = len(text) - latin_chars - cjk_chars - emoji_chars
    
    # Very conservative estimates:
    # - Latin: 1 token per 3 characters (vs typical 4:1 ratio)
    # - CJK: 2 tokens per character
    # - Emoji: 3 tokens per character
    # - Other: 1 token per 2 characters
    token_estimate = (
        (latin_chars + 2) // 3 +
        cjk_chars * 2 +
        emoji_chars * 3 +
        (other_chars + 1) // 2
    )
    
    # Also calculate based on words (for texts that are primarily words)
    word_count = len(text.split())
    word_based_estimate = int(word_count * 1.6) + 5  # About 1.6 tokens per word + overhead
    
    # Take the maximum of the two approaches
    result = max(token_estimate, word_based_estimate)
    
    # Add 5% overhead to be conservative
    return int(result * 1.05) + 10