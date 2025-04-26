import json
import os

# Load configuration from external JSON file
CONFIG_PATH = os.path.join("resources", "wood_price.json")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

grade_prices = config["grade_prices"]
CURRENCY = config["currency"]

def get_price(teak_grade):
    return grade_prices.get(teak_grade.upper(), 0)

def get_currency():
    return CURRENCY



