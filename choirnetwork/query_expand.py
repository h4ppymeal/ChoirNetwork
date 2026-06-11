"""Expand sermon and Bible narrative queries into searchable thematic text."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# Curated sermon / Bible narrative → hymn-theme phrases (offline, no API required).
SERMON_TOPIC_THEMES: dict[str, str] = {
    "ruth and naomi": (
        "loyalty devotion faithfulness providence redemption widow gleaning "
        "harvest kinsman redeemer steadfast love kindness bereavement trust god"
    ),
    "prodigal son": (
        "forgiveness repentance mercy fathers love return home grace reconciliation "
        "sin restored welcome compassion waiting father"
    ),
    "good samaritan": (
        "neighbor mercy compassion kindness love one another help stranger "
        "good deeds serve others charity"
    ),
    "daniel in the lions den": (
        "faith courage prayer deliverance protection trust god trial "
        "faithfulness standing firm persecution"
    ),
    "david and goliath": (
        "faith courage trust god victory battle strength weakness "
        "deliverance standing firm against evil"
    ),
    "moses and the red sea": (
        "deliverance salvation freedom trust god miracle power "
        "guidance exodus redemption passage through trial"
    ),
    "noahs ark": (
        "faith obedience salvation covenant trust god judgment mercy "
        "protection new beginning promise rainbow"
    ),
    "joseph and his brothers": (
        "forgiveness providence jealousy reconciliation trust god "
        "suffering purpose faithfulness hope restored family"
    ),
    "esther": (
        "courage providence deliverance faithfulness purpose queen "
        "stand firm save people trust god timing"
    ),
    "jonah": (
        "obedience repentance mercy second chance mission call "
        "forgiveness running from god compassion nations"
    ),
    "holy communion": (
        "lords supper bread wine cup remembrance body blood sacrifice "
        "cross table communion memorial feast covenant"
    ),
    "lords supper": (
        "holy communion bread wine cup remembrance body blood sacrifice "
        "cross table memorial feast broken body shed blood"
    ),
    "the cross": (
        "crucifixion sacrifice atonement redemption blood salvation "
        "suffering saviour died for sin calvary"
    ),
    "resurrection": (
        "risen alive easter victory over death empty tomb hope "
        "conquered grave eternal life triumph"
    ),
    "gods grace": (
        "grace mercy unmerited favour forgiveness kindness love "
        "undeserved gift salvation free"
    ),
    "walking in faith": (
        "faith trust obedience follow jesus daily walk discipleship "
        "step by step reliance on god"
    ),
    "prayer": (
        "pray communion with god supplication intercession kneel "
        "talk to god seek guidance sweet hour of prayer"
    ),
    "repentance": (
        "repent turn from sin confession forgiveness mercy "
        "contrite heart return to god sorrow for sin"
    ),
    "baptism": (
        "born again water burial resurrection new life wash "
        "cleansing covenant obedience immersion"
    ),
    "second coming": (
        "return of christ coming again watch ready judgment "
        "hope glory trumpet meet the lord"
    ),
    "gods providence": (
        "providence care guidance watchful eye all things work "
        "trust hardship provision faithful hand leads"
    ),
    "comfort in suffering": (
        "comfort sorrow grief trial affliction peace hope "
        "god cares carry burden strength in weakness"
    ),
    "evangelism": (
        "gospel witness tell the story share faith mission "
        "salvation proclaim good news harvest souls"
    ),
    "thanksgiving": (
        "gratitude praise thanks count blessings grateful heart "
        "give thanks rejoice blessing"
    ),
    "water into wine": (
        "miracle wedding Cana first sign glory obedience faith "
        "transformation water wine feast celebration belief"
    ),
    "wedding at cana": (
        "miracle wedding Cana first sign glory obedience faith "
        "water wine feast celebration belief transformation"
    ),
}

_LLM_SYSTEM_PROMPT = (
    "You expand sermon or Bible story titles into short thematic phrases "
    "that might appear in Christian hymn lyrics. "
    "Return only a space-separated list of 8-12 theme words and phrases, "
    "no explanation."
)

_LLM_VALIDATE_SYSTEM_PROMPT = (
    "You refine hymn-search theme keywords for a specific sermon or Bible story.\n"
    "You will receive the user's query and candidate keywords from a curated list.\n"
    "Your job:\n"
    "- KEEP only candidates that genuinely fit THIS specific story (not generic hymn words)\n"
    "- DROP mismatched or overly generic candidates\n"
    "- ADD 2-5 additional theme words that fit hymn lyrics for this story\n"
    "Return ONLY a space-separated list of 8-12 theme words. No explanation."
)

DEFAULT_LLM_MODEL = "gpt-4o-mini"
LLM_FALLBACK_MODEL = "gpt-4o-mini"
GPT5_MAX_COMPLETION_TOKENS = 512
GPT4_MAX_TOKENS = 120
_LAST_LLM_ERROR: str | None = None


def default_llm_model() -> str:
    return os.environ.get("OPENAI_LLM_MODEL", DEFAULT_LLM_MODEL)


def get_last_llm_error() -> str | None:
    return _LAST_LLM_ERROR


def _normalize_topic(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _curated_expansion(query: str) -> str | None:
    normalized = _normalize_topic(query)
    if normalized in SERMON_TOPIC_THEMES:
        return SERMON_TOPIC_THEMES[normalized]

    for topic, themes in SERMON_TOPIC_THEMES.items():
        if topic in normalized or normalized in topic:
            return themes

    return None


def _chat_completion_payload(model: str, messages: list[dict[str, str]]) -> dict:
    """Build a chat payload compatible with GPT-5 and legacy GPT-4 models."""
    if model.startswith(("gpt-4", "gpt-3")):
        return {
            "model": model,
            "messages": messages,
            "max_tokens": GPT4_MAX_TOKENS,
            "temperature": 0.2,
        }
    return {
        "model": model,
        "messages": messages,
        "max_completion_tokens": GPT5_MAX_COMPLETION_TOKENS,
    }


def _extract_message_content(body: dict) -> str | None:
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _sanitize_themes(text: str) -> str:
    """Normalize LLM theme output to a single space-separated line."""
    text = text.strip().strip("\"'")
    if "\n" in text:
        text = text.split("\n", maxsplit=1)[0].strip()
    text = re.sub(r"[^\w\s'-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _llm_chat(messages: list[dict[str, str]], *, api_key: str, model: str) -> str | None:
    global _LAST_LLM_ERROR
    _LAST_LLM_ERROR = None

    models_to_try = [model]
    if model != LLM_FALLBACK_MODEL:
        models_to_try.append(LLM_FALLBACK_MODEL)

    for attempt_model in models_to_try:
        payload = json.dumps(_chat_completion_payload(attempt_model, messages)).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            _LAST_LLM_ERROR = f"HTTP {exc.code} ({attempt_model}): {detail[:300]}"
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _LAST_LLM_ERROR = f"{attempt_model}: {exc}"
            continue

        content = _extract_message_content(body)
        if content:
            sanitized = _sanitize_themes(content)
            if sanitized:
                return sanitized

        finish_reason = body.get("choices", [{}])[0].get("finish_reason")
        _LAST_LLM_ERROR = (
            f"{attempt_model}: empty content (finish_reason={finish_reason!r})"
        )

    return None


def _llm_expansion(query: str, *, api_key: str, model: str) -> str | None:
    return _llm_chat(
        [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        api_key=api_key,
        model=model,
    )


def _llm_validate_curated(
    query: str,
    curated_themes: str,
    *,
    api_key: str,
    model: str,
) -> str | None:
    """Ask the LLM which curated themes fit this story; add a few more if needed."""
    user_content = (
        f"Query: {query}\n"
        f"Candidate themes: {curated_themes}"
    )
    return _llm_chat(
        [
            {"role": "system", "content": _LLM_VALIDATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        api_key=api_key,
        model=model,
    )


def expand_query(
    query: str,
    *,
    use_llm: bool = False,
    llm_model: str | None = None,
) -> tuple[str, str | None]:
    """Return (expanded_query, expansion_source).

    expansion_source is 'curated', 'curated+llm', 'llm', or None.
    When use_llm is on and a curated topic matches, the LLM validates which
    curated keywords fit the specific story (drops generic mismatches) and
    may add a few more; it does not blindly append the full curated list.
    """
    query = query.strip()
    if not query:
        return query, None

    model = llm_model or default_llm_model()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() if use_llm else ""

    curated = _curated_expansion(query)
    if curated:
        if api_key:
            validated = _llm_validate_curated(
                query,
                curated,
                api_key=api_key,
                model=model,
            )
            if validated:
                return f"{query} {validated}", "curated+llm"
        return f"{query} {curated}", "curated"

    if api_key:
        llm_themes = _llm_expansion(query, api_key=api_key, model=model)
        if llm_themes:
            return f"{query} {llm_themes}", "llm"

    return query, None
