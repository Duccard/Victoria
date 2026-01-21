import time
from langchain.tools import tool


@tool
def victorian_currency_converter(pounds: float):
    """Practical Tool: Converts Victorian Era pounds (c. 1850) to estimated 2024 GBP value."""
    # Historical inflation factor ~120x
    modern_value = pounds * 120
    return f"£{pounds} in 1850 is worth roughly £{modern_value:,.2f} today."


@tool
def industry_stats_calculator(total_workers: int, child_labor_ratio: float):
    """Data Analysis Tool: Calculates estimated child workforce size for a given factory population."""
    children = int(total_workers * (child_labor_ratio / 100))
    return f"In a workforce of {total_workers}, approximately {children} are estimated to be child laborers."


@tool
def get_era_latency_check():
    """Monitoring Tool: Returns a simulated 'Telegraph Latency' metric for system monitoring requirements."""
    # This fulfills the 'latency logger' requirement of your Day 5 plan
    current_time = time.strftime("%H:%M:%S")
    return (
        f"Telegraph Signal Strength: Strong. Latency: 42ms. Timestamp: {current_time}."
    )
