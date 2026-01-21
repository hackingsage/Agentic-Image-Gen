def is_blocked_prompt(text: str) -> bool:
    """
    Minimal safety filter (optional).
    """
    banned = ["child sexual", "csam", "rape", "bestiality"]
    t = (text or "").lower()
    return any(b in t for b in banned)
