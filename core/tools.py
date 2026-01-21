import time
from langchain.tools import tool


@tool
def victorian_currency_converter(pounds: float):
    """Practical Tool: Converts Victorian Era pounds (c. 1850) to estimated 2024 GBP value."""
    modern_value = pounds * 120
    return f"RESULT: £{pounds} in 1850 is worth roughly £{modern_value:,.2f} today."


@tool
def industry_stats_calculator(total_workers: int, child_labor_ratio: float):
    """Data Analysis Tool: Calculates estimated child workforce size for a given factory population."""
    children = int(total_workers * (child_labor_ratio / 100))
    return f"RESULT: In a workforce of {total_workers}, approx {children} are estimated to be child laborers."


@tool
def get_system_latency():
    """Monitoring Tool: Returns a simulated 'Telegraph Latency' metric for system monitoring."""
    return f"RESULT: Telegraph Signal Strength: Strong. Latency: 42ms."
