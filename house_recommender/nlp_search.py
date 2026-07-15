"""
Natural-language search powered by OpenAI.

Turns a sentence like "3-bed flat in Dhaka under 90 lakh" into the structured
preferences dict the recommender expects, using OpenAI structured outputs so the
model must return exactly that JSON shape. Needs OPENAI_API_KEY; if it's missing,
parse_query raises NLPUnavailable and the UI falls back to the sidebar filters.
"""

import os
import json

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

MODEL = "gpt-4o-mini"

# The 8 fields the recommender understands. Each is nullable — the model returns
# null for anything the user didn't mention.
PREF_KEYS = [
    "district", "location", "house_size", "bedrooms",
    "bathrooms", "area_sqft", "budget_bdt", "max_price_per_sqft",
]


class NLPUnavailable(Exception):
    """Raised when NL search can't run (no SDK or no API key)."""


def _schema():
    def nullable(inner):
        return {"anyOf": [inner, {"type": "null"}]}

    return {
        "type": "object",
        "properties": {
            "district": nullable({"type": "string"}),
            "location": nullable({"type": "string"}),
            "house_size": nullable({"type": "string", "enum": ["Small", "Medium", "Large"]}),
            "bedrooms": nullable({"type": "integer"}),
            "bathrooms": nullable({"type": "integer"}),
            "area_sqft": nullable({"type": "number"}),
            "budget_bdt": nullable({"type": "number"}),
            "max_price_per_sqft": nullable({"type": "number"}),
        },
        "required": PREF_KEYS,
        "additionalProperties": False,
    }


def _system_prompt(districts):
    district_list = ", ".join(districts) if districts else "(none known)"
    return (
        "You extract structured house-search preferences from a user's "
        "natural-language request for real estate in Bangladesh.\n\n"
        "Rules:\n"
        "- Return null for anything the user does NOT mention. Do not guess.\n"
        "- Money is in Bangladeshi Taka (BDT). Convert spoken amounts:\n"
        "    'lakh' or 'lac' = 100,000 ; 'crore' = 10,000,000 ; 'k' = 1,000.\n"
        "  Example: '90 lakh' -> 9000000 ; '1.2 crore' -> 12000000.\n"
        "- budget_bdt is the user's maximum total budget.\n"
        "- max_price_per_sqft only if the user explicitly limits price per sqft.\n"
        "- area_sqft is the desired floor area in square feet.\n"
        "- house_size must be one of Small, Medium, Large (or null).\n"
        f"- district: if the user names a place, normalise it to one of these "
        f"known districts when it clearly matches, else null: {district_list}.\n"
        "- location: a neighbourhood/area name (e.g. Mirpur, Banani) if given.\n"
    )


def parse_query(text, districts=None):
    """Parse a natural-language query into a preferences dict.

    Raises NLPUnavailable if the SDK or API key is missing.
    """
    if OpenAI is None:
        raise NLPUnavailable("The 'openai' package is not installed.")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-REPLACE"):
        raise NLPUnavailable(
            "OPENAI_API_KEY is not set. Add your real key to "
            ".streamlit/secrets.toml (replace the placeholder)."
        )

    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _system_prompt(districts or [])},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "house_preferences",
                "strict": True,
                "schema": _schema(),
            },
        },
    )

    # The json_schema response format guarantees valid JSON in the message.
    payload = response.choices[0].message.content
    data = json.loads(payload)
    return {k: data.get(k) for k in PREF_KEYS}


if __name__ == "__main__":
    q = "3 bedroom family flat in Dhaka under 90 lakh, at least 1400 sqft, medium size"
    print(parse_query(q, ["Dhaka", "Chattogram", "Sylhet"]))
