"""Generate the 14 synthetic datasets used in the guide.

Run with Python to recreate the CSVs. Read each numbered block independently
to see how the columns and target were made.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rows = 1500
output_dir = Path(__file__).resolve().parent / "datasets"
output_dir.mkdir(parents=True, exist_ok=True)

# Seed 42 makes the data reproducible. Each block uses its own random stream.
dataset_seeds = np.random.SeedSequence(42).spawn(14)


# 01. House prices
rng = np.random.default_rng(dataset_seeds[0])
area = rng.integers(650, 3800, rows)
bedrooms = np.clip(np.rint(area / 800 + rng.normal(0.7, 0.7, rows)), 1, 6).astype(int)
bathrooms = np.clip(bedrooms + rng.integers(-2, 1, rows), 1, 5)
age = rng.integers(0, 65, rows)
distance = rng.uniform(1, 32, rows).round(2)
price = 125000.0 + 155 * area + 12500 * bedrooms + 18500 * bathrooms
price -= 1500 * age + 2900 * distance
price += rng.normal(0, 31000, rows)
frame = pd.DataFrame({
    "area_sqft": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "house_age_years": age,
    "city_centre_distance_km": distance,
    "house_price_usd": price.clip(40000).round(2),
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "city_centre_distance_km"] = np.nan
frame.to_csv(output_dir / "01_house_prices.csv", index=False, lineterminator="\n")
print("01_house_prices.csv", frame.shape)


# 02. Student placement
rng = np.random.default_rng(dataset_seeds[1])
cgpa = rng.uniform(5.5, 10, rows).round(2)
aptitude = rng.integers(35, 101, rows)
coding = rng.integers(0, 201, rows)
internship = rng.choice(["Yes", "No"], rows, p=[0.45, 0.55])
communication = rng.integers(35, 101, rows)
score = (0.72 * (cgpa - 7.8) + 0.033 * (aptitude - 70)
         + 0.011 * (coding - 95) + 0.75 * (internship == "Yes")
         + 0.028 * (communication - 70) - 0.35)
# A larger score makes Placed more likely, then a random draw chooses the label.
probability = 1 / (1 + np.exp(-score))
labels = np.where(rng.random(rows) < probability, "Placed", "Not Placed")
frame = pd.DataFrame({
    "cgpa": cgpa,
    "aptitude_score": aptitude,
    "coding_problems_solved": coding,
    "internship_experience": internship,
    "communication_score": communication,
    "placement": labels,
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "communication_score"] = np.nan
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "internship_experience"] = np.nan
frame.to_csv(output_dir / "02_student_placement.csv", index=False, lineterminator="\n")
print("02_student_placement.csv", frame.shape)


# 03. Vehicle braking distance
rng = np.random.default_rng(dataset_seeds[2])
speed = rng.uniform(15, 125, rows).round(2)
distance = 0.8 + 0.0051 * speed**2 + rng.normal(0, 1.3 + 0.045 * speed, rows)
frame = pd.DataFrame({
    "speed_kmh": speed,
    "braking_distance_m": distance.clip(0.4).round(2),
})
frame.to_csv(output_dir / "03_vehicle_braking.csv", index=False, lineterminator="\n")
print("03_vehicle_braking.csv", frame.shape)


# 04. Bank loan approval
rng = np.random.default_rng(dataset_seeds[3])
credit = rng.integers(480, 851, rows)
income = rng.integers(1800, 14500, rows)
amount = rng.integers(2000, 58000, rows)
debt = rng.uniform(0.05, 0.7, rows).round(3)
employment = rng.integers(0, 26, rows)
score = (-1.45 + 2.0 * (credit >= 665) + 1.55 * (debt < 0.37)
         + 1.35 * (amount / income < 6.0) + 0.75 * (employment >= 3)
         - 1.8 * ((credit < 620) & (debt >= 0.40)) - 1.25 * (debt >= 0.50))
# A larger score makes Approved more likely, then a random draw chooses the label.
probability = 1 / (1 + np.exp(-score))
labels = np.where(rng.random(rows) < probability, "Approved", "Rejected")
frame = pd.DataFrame({
    "credit_score": credit,
    "monthly_income_usd": income,
    "loan_amount_usd": amount,
    "debt_to_income_ratio": debt,
    "employment_years": employment,
    "loan_status": labels,
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "debt_to_income_ratio"] = np.nan
frame.to_csv(output_dir / "04_loan_approval.csv", index=False, lineterminator="\n")
print("04_loan_approval.csv", frame.shape)


# 05. Used car prices
rng = np.random.default_rng(dataset_seeds[4])
age = rng.integers(1, 19, rows)
kilometres = (age * rng.uniform(4500, 18000, rows) + rng.uniform(500, 26000, rows)).round().astype(int)
engine = rng.choice([1.0, 1.2, 1.6, 2.0, 2.5, 3.0], rows)
owners = np.minimum(1 + rng.poisson(age / 7), 6)
accident = rng.choice(["None", "Minor", "Major"], rows, p=[0.73, 0.21, 0.06])
price = (21000 + 4700 * engine) * np.exp(-0.085 * age - kilometres / 540000)
price -= 430 * (owners - 1) + np.select([accident == "Minor", accident == "Major"], [1050, 3400])
price += rng.normal(0, 1500, rows)
frame = pd.DataFrame({
    "vehicle_age_years": age,
    "kilometres_driven": kilometres,
    "engine_capacity_l": engine,
    "previous_owners": owners,
    "accident_history": accident,
    "resale_price_usd": price.clip(800).round(2),
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "engine_capacity_l"] = np.nan
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "accident_history"] = np.nan
frame.to_csv(output_dir / "05_used_car_prices.csv", index=False, lineterminator="\n")
print("05_used_car_prices.csv", frame.shape)


# 06. Customer churn
rng = np.random.default_rng(dataset_seeds[5])
tenure = rng.integers(1, 81, rows)
bill = rng.uniform(22, 135, rows).round(2)
tickets = rng.poisson(2.2, rows)
contract = rng.choice(["Monthly", "Annual", "Two-year"], rows, p=[0.54, 0.30, 0.16])
autopay = rng.choice(["Yes", "No"], rows, p=[0.6, 0.4])
score = (-2.55 + 1.25 * (contract == "Monthly") + 0.67 * (autopay == "No")
         + 0.40 * tickets + 0.02 * (bill - 70) + 0.8 * (tenure < 12)
         + 1.2 * ((contract == "Monthly") & (tickets >= 5)))
# A larger score makes Churn more likely, then a random draw chooses the label.
probability = 1 / (1 + np.exp(-score))
labels = np.where(rng.random(rows) < probability, "Churn", "No Churn")
frame = pd.DataFrame({
    "tenure_months": tenure,
    "monthly_bill_usd": bill,
    "support_tickets": tickets,
    "contract_type": contract,
    "automatic_payment": autopay,
    "churn": labels,
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "monthly_bill_usd"] = np.nan
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "contract_type"] = np.nan
frame.to_csv(output_dir / "06_customer_churn.csv", index=False, lineterminator="\n")
print("06_customer_churn.csv", frame.shape)


# 07. Crop yield
rng = np.random.default_rng(dataset_seeds[6])
rain = rng.uniform(230, 1050, rows).round(1)
temp = rng.uniform(15, 36, rows).round(1)
ph = rng.uniform(4.9, 8.3, rows).round(2)
fertilizer = rng.uniform(25, 220, rows).round(1)
irrigation = rng.uniform(0, 70, rows).round(1)
water = rain + 4 * irrigation
yield_t = (6.2 - 0.000008 * (water - 745)**2 - 0.016 * (temp - 25)**2
           - 0.63 * (ph - 6.5)**2 + 2.0 * (1 - np.exp(-fertilizer / 85))
           + 0.009 * irrigation * (temp > 29) + rng.normal(0, 0.46, rows))
frame = pd.DataFrame({
    "rainfall_mm": rain,
    "temperature_c": temp,
    "soil_ph": ph,
    "fertilizer_kg_ha": fertilizer,
    "irrigation_hours": irrigation,
    "crop_yield_t_ha": yield_t.clip(0.25).round(2),
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "soil_ph"] = np.nan
frame.to_csv(output_dir / "07_crop_yield.csv", index=False, lineterminator="\n")
print("07_crop_yield.csv", frame.shape)


# 08. Fruit classification
rng = np.random.default_rng(dataset_seeds[7])
fruit = rng.choice(["Apple", "Orange", "Grape"], rows)
centres = np.array([[155, 13, 6.6, 52], [171, 11, 5.0, 69], [8, 17, 3.6, 44]])
row = pd.Series(fruit).map({"Apple": 0, "Orange": 1, "Grape": 2}).to_numpy()
features = rng.normal(centres[row], [33, 2.1, 0.95, 13])
small_grapes = fruit == "Grape"
features[small_grapes, 0] = rng.normal(8, 2.1, small_grapes.sum())
features = np.clip(features, [2, 4, 1, 0], [300, 25, 10, 100]).round(2)
frame = pd.DataFrame(features, columns=["weight_g", "sweetness_brix", "firmness_score", "colour_score"])
frame["fruit"] = fruit

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "firmness_score"] = np.nan
frame.to_csv(output_dir / "08_fruit_classification.csv", index=False, lineterminator="\n")
print("08_fruit_classification.csv", frame.shape)


# 09. Food delivery time
rng = np.random.default_rng(dataset_seeds[8])
distance = rng.uniform(0.5, 15, rows).round(2)
traffic = rng.integers(1, 6, rows)
preparation = rng.uniform(5, 32, rows).round(1)
weather = rng.integers(0, 4, rows)
experience = rng.uniform(0, 7, rows).round(1)
minutes = (preparation + 2.2 * distance + 0.47 * traffic * distance
           + 1.5 * weather**1.5 - 2.3 * np.log1p(experience) + rng.normal(0, 4.3, rows))
frame = pd.DataFrame({
    "delivery_distance_km": distance,
    "traffic_level": traffic,
    "preparation_time_min": preparation,
    "weather_severity": weather,
    "rider_experience_years": experience,
    "delivery_time_min": minutes.clip(7).round(1),
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "rider_experience_years"] = np.nan
frame.to_csv(output_dir / "09_food_delivery.csv", index=False, lineterminator="\n")
print("09_food_delivery.csv", frame.shape)


# 10. Student exam results
rng = np.random.default_rng(dataset_seeds[9])
attendance = rng.uniform(37, 100, rows).round(1)
internal = rng.uniform(20, 100, rows).round(1)
assignment = (0.28 * internal + rng.uniform(15, 78, rows)).clip(20, 100).round(1)
grade = (0.58 * internal + 0.42 * assignment + 0.2 * (attendance - 70)
         - 19 * (attendance < 62) + rng.normal(0, 8.0, rows))
frame = pd.DataFrame({
    "attendance_pct": attendance,
    "internal_marks": internal,
    "assignment_score": assignment,
    "exam_result": np.where(grade >= 57, "Pass", "Fail"),
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "assignment_score"] = np.nan
frame.to_csv(output_dir / "10_exam_results.csv", index=False, lineterminator="\n")
print("10_exam_results.csv", frame.shape)


# 11. Concrete strength
rng = np.random.default_rng(dataset_seeds[10])
cement = rng.uniform(150, 475, rows).round(1)
water = rng.uniform(135, 235, rows).round(1)
ash = rng.uniform(0, 145, rows).round(1)
slag = rng.uniform(0, 175, rows).round(1)
aggregate = rng.uniform(1500, 1930, rows).round(1)
plasticizer = rng.uniform(0, 18, rows).round(2)
age = rng.choice([3, 7, 14, 28, 56, 90], rows, p=[0.08, 0.15, 0.12, 0.36, 0.16, 0.13])
binder_ratio = (cement + 0.42 * ash + 0.60 * slag) / water
strength = (5 + 12 * binder_ratio * (1 - np.exp(-age / 16))
            + 4.8 * np.log1p(age / 7) + 0.42 * plasticizer
            - 0.000045 * (aggregate - 1720)**2 + rng.normal(0, 3.4, rows))
frame = pd.DataFrame({
    "cement_kg_m3": cement,
    "water_kg_m3": water,
    "fly_ash_kg_m3": ash,
    "blast_furnace_slag_kg_m3": slag,
    "aggregates_kg_m3": aggregate,
    "superplasticizer_kg_m3": plasticizer,
    "curing_age_days": age,
    "compressive_strength_mpa": strength.clip(5).round(2),
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "fly_ash_kg_m3"] = np.nan
frame.to_csv(output_dir / "11_concrete_strength.csv", index=False, lineterminator="\n")
print("11_concrete_strength.csv", frame.shape)


# 12. SMS spam
rng = np.random.default_rng(dataset_seeds[11])


SMS_TEMPLATES = {
    "Ham": [
        "{name}, can we meet at {place} at {time}?",
        "Please send me the {topic} notes before {time}, {name}.",
        "I left the {topic} folder near {place}. Can you pick it up, {name}?",
        "Reminder for our {topic} practice at {time}.",
        "{name}, tickets for {place} are ready. Meet me at {time}.",
        "I'll call when I get to {place}, {name}.",
        "Are you free to help with {topic} at {time}, {name}?",
        "Our group moved the {topic} meeting from {place} to {other_place}.",
        "Your library reservation for {topic} is ready at {place}, {name}. Reply YES to confirm.",
        "Thank you for helping with {topic}. Lunch at {place} at {time} tomorrow, {name}?",
        "Free tickets from the {topic} club, {name}. Claim yours at {place} before {time}.",
        "{name}, the bus to {place} leaves at {time}.",
        "Can you confirm the {topic} appointment at {time}, {name}?",
        "{name}, we still need {other_name} to reply about {topic}.",
        "Can you bring the {reward} we bought for {other_name} to {place}, {name}?",
        "Our {topic} class ends at {time}. I will wait outside {place}.",
    ],
    "Spam": [
        "Urgent {name}! Claim your free {reward}. Reply {code} at once.",
        "Congratulations {name}, you won a {reward}. Text WIN to {code}.",
        "Exclusive offer! Get a free {reward} before {time}. Reply {code}.",
        "Final reminder {name}: your unclaimed {reward} expires at {time}.",
        "Limited offer for {place} shoppers. Text {code} to claim a {reward}.",
        "Your number won our {reward} draw. Call {code} to collect, {name}.",
        "VIP member {name}, unlock your bonus {reward} today. Send {code}.",
        "Reply YES for a chance to win a {reward} before {time}. Ref {code}.",
        "Your {reward} reward is ready. Confirm collection by texting {code}.",
        "{name}, receive a guaranteed {reward}. Offer closes at {time}.",
        "Win a free {reward} for you and {other_name}. Reply by {time}.",
        "Flash bonus near {place}. Claim your {reward} with code {code}.",
        "Only minutes left to redeem your {reward}, {name}. Contact {code}.",
        "{name}, please confirm the meeting at {place} at {time}. Bring this code: {code}.",
        "Hi {name}, I left an envelope at {place}. Text me {code} when you collect it.",
        "Your {topic} reservation is ready for collection at {place}. Please reply YES with reference {code}.",
    ],
}

# Make one row per message. Each template family belongs to one class.
ham_rows = 960
spam_rows = 540
ham_repeats = np.full(16, 60)
spam_repeats = np.full(16, 33)
spam_repeats[:12] = 34

ham_templates = np.repeat(SMS_TEMPLATES["Ham"], ham_repeats)
spam_templates = np.repeat(SMS_TEMPLATES["Spam"], spam_repeats)
messages = np.concatenate([ham_templates, spam_templates])
labels = np.repeat(["Ham", "Spam"], [ham_rows, spam_rows])

template_numbers = np.char.mod("%02d", np.arange(1, 17))
ham_families = np.repeat(np.char.add("ham_", template_numbers), ham_repeats)
spam_families = np.repeat(np.char.add("spam_", template_numbers), spam_repeats)
families = np.concatenate([ham_families, spam_families])

# Draw example values and replace each placeholder in the template text.
names = ["Asha", "Noah", "Mia", "Ravi", "Lena", "Omar", "Sofia", "Kai", "Isha", "Sam"]
places = ["the library", "West Street", "the station", "the cafe", "Cedar Hall", "Riverside"]
topics = ["chemistry", "music", "statistics", "robotics", "design", "language", "history"]
rewards = ["gift card", "cash bonus", "shopping voucher", "phone upgrade", "travel credit"]

name_id = rng.integers(0, len(names), rows)
other_name_id = (name_id + rng.integers(1, len(names), rows)) % len(names)
place_id = rng.integers(0, len(places), rows)
other_place_id = (place_id + rng.integers(1, len(places), rows)) % len(places)
times = np.char.mod("%02d:", rng.integers(8, 22, rows))
times = np.char.add(times, np.char.mod("%02d", rng.integers(0, 6, rows) * 10))

# References are unique and randomly ordered across both classes.
reference = rng.choice(90_000, rows, replace=False) + 10_000
reference = reference.astype(str)
messages = np.char.replace(messages, "{name}", np.take(names, name_id))
messages = np.char.replace(messages, "{other_name}", np.take(names, other_name_id))
messages = np.char.replace(messages, "{place}", np.take(places, place_id))
messages = np.char.replace(messages, "{other_place}", np.take(places, other_place_id))
messages = np.char.replace(messages, "{topic}", rng.choice(topics, rows))
messages = np.char.replace(messages, "{time}", times)
messages = np.char.replace(messages, "{reward}", rng.choice(rewards, rows))
messages = np.char.replace(messages, "{code}", reference)
messages = np.char.add(messages, np.char.add(" [ref ", np.char.add(reference, "]")))

frame = pd.DataFrame({
    "message_text": messages,
    "template_family": families,
    "sms_label": labels,
})
frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
frame.to_csv(output_dir / "12_sms_spam.csv", index=False, lineterminator="\n")
print("12_sms_spam.csv", frame.shape)


# 13. Employee attrition
rng = np.random.default_rng(dataset_seeds[12])
satisfaction = rng.integers(1, 6, rows)
income = rng.integers(2300, 12500, rows)
years = rng.integers(0, 23, rows)
overtime = rng.choice(["Yes", "No"], rows, p=[0.32, 0.68])
promotions = np.minimum(rng.poisson(years / 8), 5)
balance = rng.integers(1, 6, rows)
score = (-1.1 - 0.49 * (satisfaction - 3) - 0.00018 * (income - 7000)
         - 0.055 * years + 0.95 * (overtime == "Yes") - 0.28 * promotions
         - 0.31 * (balance - 3) + 0.95 * ((overtime == "Yes") & (balance <= 2))
         + 0.55 * ((years >= 5) & (promotions == 0)))
# A larger score makes Leaving more likely, then a random draw chooses the label.
probability = 1 / (1 + np.exp(-score))
labels = np.where(rng.random(rows) < probability, "Leaving", "Staying")
frame = pd.DataFrame({
    "job_satisfaction": satisfaction,
    "monthly_income_usd": income,
    "years_at_company": years,
    "overtime": overtime,
    "past_promotions": promotions,
    "work_life_balance": balance,
    "attrition": labels,
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "work_life_balance"] = np.nan
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "overtime"] = np.nan
frame.to_csv(output_dir / "13_employee_attrition.csv", index=False, lineterminator="\n")
print("13_employee_attrition.csv", frame.shape)


# 14. Electricity demand
rng = np.random.default_rng(dataset_seeds[13])
temperature = rng.uniform(4, 39, rows).round(1)
humidity = rng.uniform(25, 95, rows).round(1)
hour = rng.integers(0, 24, rows)
day = rng.choice(["Weekday", "Weekend", "Holiday"], rows, p=[0.70, 0.26, 0.04])
activity = (rng.uniform(30, 83, rows) + 11 * (day == "Weekday")
            + 7 * ((hour >= 8) & (hour <= 18))).clip(0, 100).round(1)
evening_peak = 190 * np.exp(-((hour - 19) / 3.1)**2)
morning_peak = 95 * np.exp(-((hour - 8) / 2.1)**2)
demand = (470 + 1.12 * (temperature - 21)**2 + evening_peak + morning_peak
          + 4.6 * activity + 1.5 * (humidity - 58).clip(0) * (temperature > 27)
          - 65 * (day == "Holiday") + rng.normal(0, 43, rows))
frame = pd.DataFrame({
    "temperature_c": temperature,
    "humidity_pct": humidity,
    "hour_of_day": hour,
    "day_type": day,
    "industrial_activity_pct": activity,
    "electricity_demand_mw": demand.round(2),
})

# Leave 30 blanks in each selected feature for the preprocessing exercise.
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "humidity_pct"] = np.nan
missing_rows = rng.choice(rows, 30, replace=False)
frame.loc[missing_rows, "day_type"] = np.nan
frame.to_csv(output_dir / "14_electricity_demand.csv", index=False, lineterminator="\n")
print("14_electricity_demand.csv", frame.shape)
