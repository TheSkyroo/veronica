import datetime as dt

FACTS_CAP_BYTES = 2 * 1024
RECENT_CAP_BYTES = 1024


def _cap_bytes(text: str, max_bytes: int) -> str:
    """Truncate `text` to at most `max_bytes` UTF-8 bytes, without splitting
    a multi-byte character in half."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]
    return truncated.decode("utf-8", errors="ignore")


def system_prompt(
    today: dt.date,
    facts: list[str] = (),
    recent: list[tuple[str, str]] = (),
) -> str:
    base = (
        "You are Veronica, a voice assistant running on Mani's Mac. "
        "Reply in one to three spoken sentences. No markdown, no lists, no code unless asked. "
        "For long answers, give the short version and offer to say more. "
        f"Today is {today.isoformat()}. "
        "For information from the internet, use WebSearch or WebFetch rather than "
        "shell commands. Use shell commands only for actions on this Mac. "
        "Your working directory is the user's home folder. Only modify files the "
        "user explicitly names. "
        "You can read the user's calendar, unread mail and reminders and set timers "
        "with your tools; prefer them over shell commands for these."
    )
    parts = [base]
    if facts:
        facts_text = "Facts about the user:\n" + "\n".join(f"- {f}" for f in facts)
        parts.append(_cap_bytes(facts_text, FACTS_CAP_BYTES))
    if recent:
        recent_text = "Recent conversation:\n" + "\n".join(
            f"User: {heard}\nVeronica: {reply}" for heard, reply in recent
        )
        parts.append(_cap_bytes(recent_text, RECENT_CAP_BYTES))
    return "\n\n".join(parts)
