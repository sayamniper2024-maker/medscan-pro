"""
fabrication_check.py

The two-pass fabrication detection system: extracts likely company/product
names from a model's answer, checks them against the real search results
the model was given, and for anything unverified, does an independent
search to check whether it's real before flagging it.

Direct port from the Colab notebook - same logic, refined through several
rounds of real failure cases (see project history): naive word-counting
-> exact substring matching -> fuzzy matching against full archive ->
two-pass independent verification -> tighter entity extraction.
"""

import re

from agent import tavily_client


COMMON_SENTENCE_STARTERS = {
    "However", "Therefore", "If", "Because", "Since", "While", "Although",
    "This", "That", "These", "Those", "The", "A", "An", "It", "There",
    "Based", "Given", "Note", "Source", "Summary", "Table", "Item", "Key",
    "Bottom", "Overall", "In", "On", "At", "For", "With", "As", "So",
    "Additionally", "Furthermore", "Moreover", "Thus", "Hence", "Conclusion",
    "Established", "Confirmed", "Funding", "Companies", "Company", "No",
    "Class", "FDA", "All", "Many", "Most", "Some",
}

HEADER_WORDS = {
    "Landscape", "Overview", "Summary", "Analysis", "Takeaway",
    "Takeaways", "Conclusion", "Findings", "Assessment", "Outlook",
    "Recommendation", "Recommendations",
}


def extract_capitalized_entities(text):
    """Pull out likely company/product names: multi-word capitalized phrases,
    excluding common sentence-starters and report-formatting header words."""
    candidates = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){1,2}\b", text)
    filtered = []
    for c in candidates:
        words = c.split()
        if words[0] in COMMON_SENTENCE_STARTERS:
            continue
        if any(w in HEADER_WORDS for w in words):
            continue
        filtered.append(c)
    return set(filtered)


def fuzzy_in_text(entity, text):
    """Check if an entity's core words appear in the text, even if exact
    phrasing differs (e.g., 'Vinci Surgical System' vs 'da Vinci Surgical
    System')."""
    entity_words = entity.lower().split()
    text_lower = text.lower()
    return all(word in text_lower for word in entity_words)


def verify_entity_exists(entity_name):
    """
    Targeted search to check whether a named entity (company/product)
    actually exists in the real world, independent of the original
    research search results. Returns True only if the exact phrase
    appears together in the fresh results, False if not, None if the
    check itself failed (treated as "unknown", not "fabricated").
    """
    try:
        results = tavily_client.search(query=f'"{entity_name}"', max_results=2)
        result_text = " ".join([r["content"] for r in results["results"]]).lower()
        return entity_name.lower() in result_text
    except Exception:
        return None


def check_for_unsourced_claims(answer_text, block_name, full_archive=None, verify_unknowns=True):
    """
    Two-pass fabrication check:
    1. Compare entities against the actual search results (fast, free)
    2. For anything not found there, do a targeted real-world existence
       check before concluding it's fabricated (slower, costs extra searches)
    """
    warnings = []

    if block_name != "competitors":
        return warnings

    if full_archive is None:
        full_archive = []

    answer_entities = extract_capitalized_entities(answer_text)
    combined_search_text = " ".join(full_archive)

    not_in_search_results = [
        entity for entity in answer_entities if not fuzzy_in_text(entity, combined_search_text)
    ]

    if not not_in_search_results:
        return warnings

    if not verify_unknowns:
        if len(not_in_search_results) > 5:
            warnings.append(
                f"{len(not_in_search_results)} entities not found in search results "
                f"(not independently verified): {not_in_search_results[:8]}"
            )
        return warnings

    likely_fabricated = []
    unknown = []

    for entity in not_in_search_results:
        exists = verify_entity_exists(entity)
        if exists is False:
            likely_fabricated.append(entity)
        elif exists is None:
            unknown.append(entity)
        # exists is True -> confirmed real, no warning needed

    if likely_fabricated:
        warnings.append(
            f"LIKELY FABRICATION: these entities were not in the original search results "
            f"AND a follow-up existence check found no evidence they're real: {likely_fabricated}. "
            f"Strongly recommend removing or independently verifying these before trusting this section."
        )

    if unknown:
        warnings.append(
            f"UNVERIFIED (not fabrication-confirmed, just unconfirmed): {unknown}. "
            f"These weren't in the original search results, and the verification check was "
            f"inconclusive. Worth a manual look, but likely just real entities the model knew "
            f"from general knowledge rather than this search."
        )

    return warnings
