# Synthetic Dataset Dictionary

Each CSV contains 1,500 rows and a header. Every dataset has its own NumPy random stream derived from seed 42. Re-run `python generate_datasets.py` from the project folder to regenerate all 14 CSVs.

These data describe fictional cases and support learning exercises. Real-world financial, medical, safety, educational and employment decisions require separately validated evidence.

The declared types describe the recorded values. CSV is a text format, so pandas chooses in-memory dtypes when it reads each file. Integer-valued columns with blanks may load as floats. Blank fields represent missing values. In selected columns, the generator masks 30 feature values per column after computing the target. Use the notebook’s CSV settings to preserve legitimate text such as the accident category `None`.

## 01. House Price Prediction

**CSV** · [01_house_prices.csv](01_house_prices.csv)  
**Records** · 1,500  
**Target column** · `house_price_usd` (USD)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `area_sqft` | integer | ft² | Interior floor area |
| `bedrooms` | integer | count | Number of bedrooms |
| `bathrooms` | integer | count | Number of bathrooms |
| `house_age_years` | integer | years | Age of the house at sale |
| `city_centre_distance_km` | decimal | km | Distance from the city centre |
| `house_price_usd` | decimal | USD | Observed continuous outcome |

**Target generation** · Price starts at 125,000 USD. It gains 155 USD per ft², 12,500 USD per bedroom and 18,500 USD per bathroom, and loses 1,500 USD per year of age and 2,900 USD per km from the centre. Independent Gaussian noise has a standard deviation of 31,000 USD. Prices are floored at 40,000 USD. Bedroom count is related to area, and bathrooms are related to bedrooms.

## 02. Student Placement Prediction

**CSV** · [02_student_placement.csv](02_student_placement.csv)  
**Records** · 1,500  
**Target column** · `placement` (Not Placed, Placed)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `cgpa` | decimal | points out of 10 | Cumulative grade-point average |
| `aptitude_score` | integer | points out of 100 | Aptitude assessment score |
| `coding_problems_solved` | integer | count | Practice problems completed |
| `internship_experience` | string | Yes or No | Whether an internship was completed |
| `communication_score` | integer | points out of 100 | Communication assessment score |
| `placement` | string | Not Placed, Placed | Observed class label |

**Target generation** · CGPA, aptitude, coding practice and communication add to a weighted log-odds score. Completing an internship adds 0.75 to that score. A sigmoid converts the score to a probability, then a Bernoulli draw produces Placed or Not Placed. Similar profiles can have different outcomes.

## 03. Vehicle Braking Distance Prediction

**CSV** · [03_vehicle_braking.csv](03_vehicle_braking.csv)  
**Records** · 1,500  
**Target column** · `braking_distance_m` (m)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `speed_kmh` | decimal | km/h | Vehicle speed when braking begins |
| `braking_distance_m` | decimal | m | Observed continuous outcome |

**Target generation** · Distance is 0.8 + 0.0051 × speed², plus Gaussian noise with standard deviation 1.3 + 0.045 × speed in metres, floored at 0.4 m. This is an illustrative relationship without reaction distance, road conditions or vehicle-specific physics.

## 04. Bank Loan Approval

**CSV** · [04_loan_approval.csv](04_loan_approval.csv)  
**Records** · 1,500  
**Target column** · `loan_status` (Rejected, Approved)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `credit_score` | integer | score points | Synthetic credit score |
| `monthly_income_usd` | integer | USD/month | Gross monthly income |
| `loan_amount_usd` | integer | USD | Principal requested |
| `debt_to_income_ratio` | decimal | fraction | Monthly debt payments divided by monthly income |
| `employment_years` | integer | years | Years of employment |
| `loan_status` | string | Rejected, Approved | Observed class label |

**Target generation** · The log-odds score includes credit thresholds at 620 and 665, debt ratios around 0.37–0.50, an amount-to-income ratio below six, years employed and a penalty when low credit and elevated debt coincide. A Bernoulli draw adds label uncertainty. These rules are educational and do not represent lending policy.

## 05. Used Car Price Prediction

**CSV** · [05_used_car_prices.csv](05_used_car_prices.csv)  
**Records** · 1,500  
**Target column** · `resale_price_usd` (USD)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `vehicle_age_years` | integer | years | Age of the used vehicle |
| `kilometres_driven` | integer | km | Odometer reading |
| `engine_capacity_l` | decimal | litres | Engine displacement |
| `previous_owners` | integer | count | Number of previous owners |
| `accident_history` | string | None, Minor or Major | Severity of recorded accident history |
| `resale_price_usd` | decimal | USD | Observed continuous outcome |

**Target generation** · Starting value increases with engine capacity and decays exponentially with vehicle age and mileage. Previous owners and accident severity reduce price. Gaussian noise with standard deviation 1,500 USD represents unobserved factors, and resale values are floored at 800 USD.

## 06. Customer Churn Prediction

**CSV** · [06_customer_churn.csv](06_customer_churn.csv)  
**Records** · 1,500  
**Target column** · `churn` (No Churn, Churn)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `tenure_months` | integer | months | Time as a customer |
| `monthly_bill_usd` | decimal | USD/month | Monthly bill |
| `support_tickets` | integer | count | Number of recent support tickets |
| `contract_type` | string | Monthly, Annual or Two-year | Renewal term |
| `automatic_payment` | string | Yes or No | Whether automatic payment is enabled |
| `churn` | string | No Churn, Churn | Observed class label |

**Target generation** · Churn log-odds rise with monthly contracts, manual payment, tickets, larger bills and short tenure. A monthly plan with at least five tickets adds an interaction term. Outcomes are sampled from the resulting sigmoid probabilities.

## 07. Agriculture Crop Yield Prediction

**CSV** · [07_crop_yield.csv](07_crop_yield.csv)  
**Records** · 1,500  
**Target column** · `crop_yield_t_ha` (t/ha)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `rainfall_mm` | decimal | mm | Growing-season rainfall |
| `temperature_c` | decimal | °C | Average growing-season temperature |
| `soil_ph` | decimal | pH | Soil acidity |
| `fertilizer_kg_ha` | decimal | kg/ha | Fertilizer applied per hectare |
| `irrigation_hours` | decimal | hours | Seasonal operating hours of irrigation |
| `crop_yield_t_ha` | decimal | t/ha | Observed continuous outcome |

**Target generation** · Rainfall plus four times irrigation hours acts as a water index. Yield peaks near a water index of 745, 25 °C and pH 6.5. Fertilizer has diminishing returns, extra irrigation has a small hot-weather effect, and Gaussian noise with standard deviation 0.46 t/ha is added before flooring.

## 08. Fruit Classification

**CSV** · [08_fruit_classification.csv](08_fruit_classification.csv)  
**Records** · 1,500  
**Target column** · `fruit` (Apple, Grape, Orange)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `weight_g` | decimal | g | Weight of one fruit |
| `sweetness_brix` | decimal | °Bx | Soluble solids on a Brix scale |
| `firmness_score` | decimal | score from 1 to 10 | Synthetic firmness measure |
| `colour_score` | decimal | score from 0 to 100 | Synthetic colour index |
| `fruit` | string | Apple, Grape, Orange | Observed class label |

**Target generation** · Fruit labels are sampled uniformly. Each label has its own average weight, sweetness, firmness and colour. Gaussian measurement variation creates overlap, particularly for apples and oranges. Grapes have a much smaller, tighter weight distribution.

## 09. Food Delivery Time Prediction

**CSV** · [09_food_delivery.csv](09_food_delivery.csv)  
**Records** · 1,500  
**Target column** · `delivery_time_min` (min)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `delivery_distance_km` | decimal | km | Road distance for the delivery |
| `traffic_level` | integer | ordinal 1 to 5 | Increasing traffic intensity |
| `preparation_time_min` | decimal | min | Estimated preparation duration |
| `weather_severity` | integer | ordinal 0 to 3 | Increasing severity of weather |
| `rider_experience_years` | decimal | years | Rider’s delivery experience |
| `delivery_time_min` | decimal | min | Observed continuous outcome |

**Target generation** · Delivery time adds preparation, base travel time, a distance-by-traffic penalty and a nonlinear weather penalty. Rider experience reduces time with diminishing returns. Gaussian noise of 4.3 minutes is added and the target is floored at seven minutes.

## 10. Student Exam Result Prediction

**CSV** · [10_exam_results.csv](10_exam_results.csv)  
**Records** · 1,500  
**Target column** · `exam_result` (Fail, Pass)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `attendance_pct` | decimal | percent | Course sessions attended |
| `internal_marks` | decimal | points out of 100 | Internal assessment result |
| `assignment_score` | decimal | points out of 100 | Coursework score |
| `exam_result` | string | Fail, Pass | Observed class label |

**Target generation** · A weighted combination of marks and assignments, an attendance adjustment and a 19-point penalty below 62% attendance creates a latent grade. Gaussian noise of eight points is added. Grades at least 57 receive Pass. Assignment and internal scores are partially correlated.

## 11. Concrete Compressive Strength Prediction

**CSV** · [11_concrete_strength.csv](11_concrete_strength.csv)  
**Records** · 1,500  
**Target column** · `compressive_strength_mpa` (MPa)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `cement_kg_m3` | decimal | kg/m³ | Cement mass per cubic metre of mix |
| `water_kg_m3` | decimal | kg/m³ | Mix water mass per cubic metre |
| `fly_ash_kg_m3` | decimal | kg/m³ | Fly ash content per cubic metre |
| `blast_furnace_slag_kg_m3` | decimal | kg/m³ | Slag content per cubic metre |
| `aggregates_kg_m3` | decimal | kg/m³ | Combined aggregate content per cubic metre |
| `superplasticizer_kg_m3` | decimal | kg/m³ | Superplasticizer per cubic metre |
| `curing_age_days` | integer | days | Age at compressive testing |
| `compressive_strength_mpa` | decimal | MPa | Observed continuous outcome |

**Target generation** · A weighted cement, fly ash and slag total divided by water gives a binder ratio. Strength grows with that ratio and curing age with diminishing returns. Superplasticizer adds a small term, aggregate deviations reduce strength, and Gaussian noise of 3.4 MPa is added before flooring.

## 12. SMS Spam Detection

**CSV** · [12_sms_spam.csv](12_sms_spam.csv)  
**Records** · 1,500  
**Target column** · `sms_label` (Ham, Spam)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `message_text` | string | SMS text | Raw message used as the only predictive input |
| `template_family` | string | group identifier | Generator template identifier used only to create disjoint splits |
| `sms_label` | string | Ham, Spam | Observed class label |

**Target generation** · The generator creates 960 Ham and 540 Spam messages from 16 templates for each label, varying names, places, times, topics, codes and reward phrases. Each message includes a unique synthetic reference from the same shuffled sequence across both classes. The rows are shuffled before saving. The template family includes the class name and is kept entirely outside the features. Synthetic wording remains simpler than real SMS.

## 13. Employee Attrition Prediction

**CSV** · [13_employee_attrition.csv](13_employee_attrition.csv)  
**Records** · 1,500  
**Target column** · `attrition` (Staying, Leaving)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `job_satisfaction` | integer | ordinal 1 to 5 | Self-reported satisfaction level |
| `monthly_income_usd` | integer | USD/month | Gross monthly income |
| `years_at_company` | integer | years | Tenure at the company |
| `overtime` | string | Yes or No | Whether overtime is worked |
| `past_promotions` | integer | count | Promotions already earned |
| `work_life_balance` | integer | ordinal 1 to 5 | Reported work-life balance |
| `attrition` | string | Staying, Leaving | Observed class label |

**Target generation** · Lower satisfaction, income, tenure, promotion counts and work-life balance increase a log-odds score. Overtime increases it, especially when work-life balance is low. Tenure of at least five years without a promotion adds another term. The label is sampled from the sigmoid probability.

## 14. Electricity Demand Prediction

**CSV** · [14_electricity_demand.csv](14_electricity_demand.csv)  
**Records** · 1,500  
**Target column** · `electricity_demand_mw` (MW)

| Column | Type | Units or values | Meaning |
| --- | --- | --- | --- |
| `temperature_c` | decimal | °C | Outdoor temperature |
| `humidity_pct` | decimal | percent | Relative humidity |
| `hour_of_day` | integer | hour 0 to 23 | Hour in local time |
| `day_type` | string | Weekday, Weekend or Holiday | Calendar day category |
| `industrial_activity_pct` | decimal | percent | Synthetic industrial activity index |
| `electricity_demand_mw` | decimal | MW | Observed continuous outcome |

**Target generation** · Demand starts at 470 MW and adds squared temperature deviation from 21 °C, Gaussian-shaped morning and evening peaks, industrial activity, and a hot-weather humidity interaction. Holidays subtract 65 MW and Gaussian noise with standard deviation 43 MW is added. Industrial activity depends partly on hour and day type. Rows are independent snapshots without timestamps.

