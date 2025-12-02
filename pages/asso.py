import streamlit as st
import pandas as pd
import itertools
import math
from collections import Counter

st.title(" Association Rule Mining - Apriori")

df = pd.read_csv("domestic_city_processed.csv")

def pax_level(x):
    if x < 2000: return "Pax_Low"
    elif x <= 10000: return "Pax_Medium"
    else: return "Pax_High"

def freight_level(x):
    if x==0: return "Freight_zero"
    elif x < 2: return "Freight_Low"
    elif x <= 50: return "Freight_Medium"
    else: return "Freight_High"

def mail_level(x):
    if x==0: return "Mail_zero"
    elif x <=2: return "Mail_Low"
    elif x <= 11: return "Mail_Medium"
    else: return "Mail_High"

def pax_to_level(x):
    if x < 1200: return "PaxTo_Low"
    elif x <= 6000: return "PaxTo_Medium"
    else: return "PaxTo_High"

def pax_from_level(x):
    if x < 1200: return "PaxFrom_Low"
    elif x <= 6000: return "PaxFrom_Medium"
    else: return "PaxFrom_High"

def freight_to_level(x):
    if x==0: return "FreightTo_zero"
    elif x < 0.75: return "FreightTo_Low"
    elif x <= 15: return "FreightTo_Medium"
    else: return "FreightTo_High"

def freight_from_level(x):
    if x==0: return "FreightFrom_zero"
    elif x < 0.75: return "FreightFrom_Low"
    elif x <= 20: return "FreightFrom_Medium"
    else: return "FreightFrom_High"

def mail_to_level(x):
    if x==0: return "MailTo_zero"
    elif x <=1.5: return "MailTo_Low"
    elif x <= 7: return "MailTo_Medium"
    else: return "MailTo_High"

def mail_from_level(x):
    if x==0: return "MailFrom_zero"
    if x <=1.5: return "MailFrom_Low"
    elif x <= 7: return "MailFrom_Medium"
    else: return "MailFrom_High"

df["Pax_Level"] = df["total_passengers"].apply(pax_level)
df["Freight_Level"] = df["total_freight"].apply(freight_level)
df["Mail_Level"] = df["total_mail"].apply(mail_level)

df["PaxTo_Level"] = df["paxtocity2"].apply(pax_to_level)
df["PaxFrom_Level"] = df["paxfromcity2"].apply(pax_from_level)
df["FreightTo_Level"] = df["freighttocity2"].apply(freight_to_level)
df["FreightFrom_Level"] = df["freightfromcity2"].apply(freight_from_level)
df["MailTo_Level"] = df["mailtocity2"].apply(mail_to_level)
df["MailFrom_Level"] = df["mailfromcity2"].apply(mail_from_level)

df["Month"] = df["month"].astype(int).map({
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
})

#basket
transactions = []
for _, row in df.iterrows():
    t = [
        f"Origin={row['city1']}",
        f"Dest={row['city2']}",
        f"Month={row['Month']}",
        row["Pax_Level"],
        row["Freight_Level"],
        row["Mail_Level"],
        row["PaxTo_Level"],
        row["PaxFrom_Level"],
        row["FreightTo_Level"],
        row["FreightFrom_Level"],
        row["MailTo_Level"],
        row["MailFrom_Level"]
    ]
    transactions.append(t)

n = len(transactions)
min_support = 0.01
min_support_count = max(1, math.ceil(min_support * n))
min_confidence = 0.4

item_counts = Counter()
for t in transactions:
    item_counts.update(set(t))

freq1 = {i for i,c in item_counts.items() if c >= min_support_count}

pairs = list(itertools.combinations(sorted(freq1), 2))
pair_count = Counter()

for t in transactions:
    s = set(t)
    for a, b in pairs:
        if a in s and b in s:
            pair_count[(a,b)] += 1

rules = []
for (a,b), cnt in pair_count.items():
    if cnt >= min_support_count:
        support = cnt / n

        conf_ab = cnt / item_counts[a]
        conf_ba = cnt / item_counts[b]

        lift_ab = conf_ab / (item_counts[b] / n)
        lift_ba = conf_ba / (item_counts[a] / n)

        if conf_ab >= min_confidence:
            rules.append([a, b, support, cnt, conf_ab, lift_ab])

        if conf_ba >= min_confidence:
            rules.append([b, a, support, cnt, conf_ba, lift_ba])

rules_df = pd.DataFrame(
    rules,
    columns=["Antecedent","Consequent","Support","SupportCount","Confidence","Lift"]
).sort_values(by=["Lift","Confidence"], ascending=False)

sorted_rules = rules_df.sort_values(by=["Lift", "Confidence", "Support"], ascending=False)


seasonal_rules = sorted_rules[
    sorted_rules["Antecedent"].str.startswith("Month=")
]

traffic_corr_rules = sorted_rules[
    (sorted_rules["Antecedent"].str.contains("Pax")) &
    (sorted_rules["Consequent"].str.contains("Freight") |
     sorted_rules["Consequent"].str.contains("Mail"))
].head(50)

imbalance_rules = sorted_rules[
    (sorted_rules["Antecedent"].str.startswith("Origin=") |
     sorted_rules["Antecedent"].str.startswith("Dest=")) &
    (sorted_rules["Consequent"].str.contains("PaxTo") |
     sorted_rules["Consequent"].str.contains("PaxFrom"))
].head(50)


tab1, tab2, tab3, tab4 = st.tabs([
    "Route Profile Rules",
    "Seasonality Rules",
    "Traffic Correlation Rules",
    "Traffic Imbalance Rules"
])

with tab1:
    st.header("Overview")
    st.markdown("""

    This page applies **Association Rule Mining** to airline domestic traffic data using a 
    market-basket approach similar to Amazon’s recommendation system.

    Each row of the dataset is converted into a **transaction basket** containing items such as:

    - `Origin=<city>`
    - `Dest=<city>`
    - `Month=<month>`
    - `Pax_Level`, `Freight_Level`, `Mail_Level` (Low / Medium / High)
    - Directional levels like `PaxTo_`, `PaxFrom_`, `FreightTo_`, `FreightFrom_`, `MailTo_`, `MailFrom_`

    To make analysis useful, numeric values are converted into categories using simple fixed 
    thresholds (e.g., Passenger > 5000 → **Pax_High**).

    Based on these baskets, we generate association rules and display insights across three major
    categories:

    ---

    ### ** Seasonality Rules**
    Identify months where specific routes show traffic spikes  
    (e.g., `Month=Dec → Pax_High`).

    ### **Traffic Correlation Rules**
    Show how passenger, freight, and mail volumes relate  
    (e.g., `Pax_High → Freight_Medium`).

    ### **Traffic Imbalance Rules**
    Highlight one-directional demand  
    (e.g., `Origin=LKO → PaxTo_High & PaxFrom_Low`).


    """)

    st.dataframe(sorted_rules)

with tab2:
    st.header("Seasonality Rules (Top 50)")
    month_list = ["All","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    selected_month = st.selectbox("Select Month", month_list)

    if selected_month == "All":
        seasonal_display = seasonal_rules     
    else:
        seasonal_display = seasonal_rules[
            seasonal_rules["Antecedent"] == f"Month={selected_month}"
        ]

    st.subheader(f"Seasonality Rules for: {selected_month}")
    st.dataframe(seasonal_display.head(50))

    st.markdown("""
    ### May
    Support = 0.0387 means: 3.87 of all transactions (rows) show both Month=May and FreightTo_Low.
    Confidence answers:
    When Month=May occurs, how often is the consequent also true?
    
    Confidence = 0.6541 means:
    When it is May, 65.4 of the time FreightTo is Low.
    
    Lift compares:Observed relationship vs. Random chance
    
    Lift = 1 → No association
    Lift > 1 → Positive correlation
    Lift < 1 → Negative correlation
    
    These all exceed 1, meaning:
    Low freight and mail levels happen more often in May than would be expected randomly.
                
    May is likely a passenger-dominated month (vacation time before monsoon).
    Airlines should increase passenger seats and reduce cargo belly usage.

    """)

with tab3:
    st.header("Traffic Correlation Rules (Top 50)")
    st.dataframe(traffic_corr_rules)
    st.markdown("""
    ### High Passenger Traffic is Strongly Linked to High Cargo Traffic
    When passenger demand is high, freight and mail volumes are also high
    Lift values between 2.5 and 3.0 show an extremely strong correlation.
    High freight/mail is three times more likely when passenger traffic is high
    compared to random chance.
                
    ###Pax_Low → Freight_Zero is also highly correlated
    Small aircraft - Lower frequency - Zero or minimal cargo operations
    """)

with tab4:
    st.header("Traffic Imbalance Rules (Top 50)")
    st.dataframe(imbalance_rules)
    st.markdown("""
    ### Delhi: High inbound AND outbound traffic imbalance
    If a route ends in Delhi, then:
    PaxFrom_High = Many people return / depart from Delhi
    PaxTo_High = Many people are flying into Delhi
    This shows two-way high demand, but more strongly inbound → DELHI is a hub city.
    
    ###Indore: Medium-range imbalance
    Indore shows medium-level two-way traffic.
                
    ###Coimbatore: Strong outbound imbalance (Weak incoming)
    Outbound demand exists (people leaving Coimbatore)
    Inbound demand is weaker
    Directional imbalance
    """)
