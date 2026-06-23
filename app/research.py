"""
research.py

The research blocks (market sizing, competitors, regulatory, strategic
signals), the pipeline runner that executes them in sequence, and the
final report assembly (executive summary, recommendations, full report
text).

Direct port from the Colab notebook.
"""

import time

from agent import run_agent, client, MODEL_NAME, TOOL_SYSTEM_PROMPT, full_search_archive, MOCK_MODE
from fabrication_check import check_for_unsourced_claims


RESEARCH_BLOCKS = {
    "market_sizing": """Research the global market for: {device_concept}, targeting: {indication}.
Find: total addressable market (TAM) in USD with year and source, growth rate (CAGR),
market maturity stage, and 2-3 key market drivers and barriers.
Cite every figure with [Number] + [Source] + [Year]. If data on this specific niche
doesn't exist, search for the closest parent market category instead and say so explicitly.""",

    "competitors": """Research the competitive landscape for: {device_concept}, targeting: {indication}.

CRITICAL RULE: Only name a company if you can see it explicitly mentioned in your search
results below. Do NOT invent, infer, or guess company names, funding figures, or product
names under any circumstances - even plausible-sounding ones. If your searches do not
surface specific named companies working on this exact niche, you MUST say so explicitly:
"No specific companies found building this exact niche product. The closest related
players in the broader [parent category] market are: [list only companies you actually
found in search results, with source]."

For each company you DO name, you must be able to point to which search result mentioned
them. If you cannot, do not include them.

Identify named companies building or selling related technology, separating established
incumbents from emerging/startup players, with funding/revenue figures ONLY if found in
search results.""",

    "regulatory": """Research the regulatory pathway for: {device_concept}, targeting: {indication}.
Find: likely FDA classification (Class I/II/III) and pathway (510k vs PMA) based on
comparable cleared devices, relevant FDA guidance documents if findable, and any
reimbursement/CPT code context. Note this is informational research only, not regulatory
or legal advice - frame findings as 'likely pathway based on comparable devices.'""",

    "strategic_signals": """Research recent strategic activity related to: {device_concept},
targeting: {indication}. Find: any M&A activity, funding rounds (Series B+), or active
clinical trials in the last 24 months related to this device category. If this is too
niche a category for direct hits, search the closest broader category and say so."""
}


REPORT_ASSEMBLY_PROMPT = """You are assembling a final market research report for a medical device concept.

DEVICE CONCEPT: {device_concept}
TARGET INDICATION: {indication}

You have been given four completed research sections below: Market Sizing, Competitive
Landscape, Regulatory & Reimbursement, and Strategic Signals. Your job is NOT to research
further - just synthesize what's already here into two new pieces:

1. EXECUTIVE SUMMARY (max 300 words): market opportunity statement, competitive intensity
   rating (1-10 with one-sentence rationale), top 3 strategic implications, recommended
   immediate next actions.

2. STRATEGIC RECOMMENDATIONS (5-7 recommendations): for each, format as:
   [Recommendation] -> [Rationale] -> [Risk if ignored] -> [Suggested Timeline]

Base both ONLY on the research provided below - do not introduce new facts, company names,
or figures that aren't already present in these sections. If the research has gaps, your
recommendations can acknowledge that (e.g., "commission primary research on X").

=== MARKET SIZING ===
{market_sizing}

=== COMPETITIVE LANDSCAPE ===
{competitors}

=== REGULATORY & REIMBURSEMENT ===
{regulatory}

=== STRATEGIC SIGNALS ===
{strategic_signals}
"""


def run_full_research(device_concept, indication, blocks=None, pause_seconds=5):
    """
    Runs all research blocks in sequence, collecting results and
    fabrication warnings for each.

    Returns: (results_dict, warnings_dict)
    """
    if blocks is None:
        blocks = RESEARCH_BLOCKS

    global full_search_archive
    full_search_archive.clear()

    results = {}
    warnings_log = {}

    for block_name, block_prompt in blocks.items():
        print(f"\n{'=' * 50}")
        print(f"RUNNING BLOCK: {block_name}")
        print(f"{'=' * 50}\n")

        block_start_index = len(full_search_archive)

        filled_prompt = block_prompt.format(
            device_concept=device_concept,
            indication=indication,
        )

        answer, search_text_list = run_agent(
            filled_prompt,
            system_prompt=TOOL_SYSTEM_PROMPT,
            max_searches=3,
            debug=True,
        )

        results[block_name] = answer

        block_archive_slice = full_search_archive[block_start_index:]
        warnings_log[block_name] = check_for_unsourced_claims(
            answer, block_name, block_archive_slice
        )

        print(f"\n--- {block_name} COMPLETE ---")
        if warnings_log[block_name]:
            for w in warnings_log[block_name]:
                print(w)
        print()

        print(f"Pausing {pause_seconds}s before next block...")
        time.sleep(pause_seconds)

    return results, warnings_log


def assemble_executive_summary_and_recommendations(device_concept, indication, results):
    filled_prompt = REPORT_ASSEMBLY_PROMPT.format(
        device_concept=device_concept,
        indication=indication,
        market_sizing=results["market_sizing"],
        competitors=results["competitors"],
        regulatory=results["regulatory"],
        strategic_signals=results["strategic_signals"],
    )

    if MOCK_MODE:
        return (
            "**MOCK EXECUTIVE SUMMARY**\n\nThis is placeholder summary/recommendations text "
            "generated in MOCK_MODE for testing the pipeline without using real API tokens.\n\n"
            "**MOCK STRATEGIC RECOMMENDATIONS**\n\n1. Mock recommendation -> Mock rationale -> "
            "Mock risk -> Mock timeline."
        )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": filled_prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate summary/recommendations: {e}"


def assemble_final_report(device_concept, indication, results, summary_and_recs, warnings_log):
    report = f"""# MedScan Pro — Market Intelligence Report

**Device Concept:** {device_concept}
**Target Indication:** {indication}
**Research Depth:** Deep Dive
**Generated:** {time.strftime("%Y-%m-%d")}

---

{summary_and_recs}

---

## SECTION 1: MARKET OVERVIEW

{results['market_sizing']}

---

## SECTION 2: COMPETITIVE LANDSCAPE

{results['competitors']}

---

## SECTION 3: REGULATORY & REIMBURSEMENT

{results['regulatory']}

---

## SECTION 4: STRATEGIC SIGNALS

{results['strategic_signals']}

---

## APPENDIX: CONFIDENCE RATINGS & DATA GAP FLAGS

"""

    for block_name in results.keys():
        report += f"**{block_name.replace('_', ' ').title()}:** "
        if warnings_log.get(block_name):
            report += "Flagged for review — " + " | ".join(warnings_log[block_name]) + "\n\n"
        else:
            report += (
                "No fabrication flags raised. Note: even unflagged sections should be "
                "spot-checked against original sources before use in investor or "
                "regulatory contexts.\n\n"
            )

    report += """
---

**Disclaimer:** This report was generated by an AI research agent (MedScan Pro) using
automated web search and large language model synthesis. It is intended for early-stage
market exploration only. It is NOT regulatory, legal, or financial advice. All figures,
regulatory pathway assessments, and competitive claims should be independently verified
before being used in investor materials, regulatory submissions, or strategic decisions.
"""

    return report


def generate_full_report(device_concept, indication):
    """
    Convenience function that runs the entire pipeline end-to-end:
    research -> summary/recommendations -> final assembled report.
    This is the single function the web app will eventually call.
    """
    results, warnings_log = run_full_research(device_concept, indication)
    summary_and_recs = assemble_executive_summary_and_recommendations(
        device_concept, indication, results
    )
    final_report = assemble_final_report(
        device_concept, indication, results, summary_and_recs, warnings_log
    )
    return final_report
