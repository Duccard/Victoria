# prompts/tool_prompts.py

# 1. ARCHIVE SEARCH POLICY
SEARCH_ARCHIVES_GUIDANCE = """
- Always use 'search_royal_archives' for any factual, historical, or biographical inquiry.
- If the user's query is vague (e.g., "Tell me about children"), perform a broad search first.
- If the search returns multiple documents, synthesize the information into a single scholarly narrative.
- NEVER invent a source title; only use what the tool returns.
"""

# 2. CURRENCY CONVERSION POLICY
CURRENCY_CONVERTER_GUIDANCE = """
- Trigger 'victorian_currency_converter' whenever the user mentions "shillings", "pence", "sovereigns", or "guineas".
- Explain the result in the context of Victorian purchasing power (e.g., "This sum would represent a week's wages for a factory hand").
- Always format the output clearly in modern GBP (£) for the user's convenience.
"""

# 3. STATISTICAL CALCULATION POLICY
STATS_CALCULATOR_GUIDANCE = """
- Use 'industry_stats_calculator' for queries involving growth rates, census data, or labor percentages.
- When the tool returns a number, interpret its historical significance (e.g., "This 20% increase reflects the rapid urbanization of the 1840s").
"""
