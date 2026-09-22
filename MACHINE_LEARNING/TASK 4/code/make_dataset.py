"""
make_dataset.py
----------------
Generates a sample SMS spam/ham dataset for Task 4, in the same format
as the classic UCI/Kaggle "SMS Spam Collection" dataset referenced by
CodSoft (columns: label, message).

NOTE: The real dataset requires downloading from Kaggle, which is not
reachable in this environment, so a representative synthetic dataset
is generated here with realistic spam/ham message patterns so the
pipeline in train_model.py runs end-to-end.

TO USE THE REAL DATASET INSTEAD:
1. Download spam.csv from the CodSoft task link (Kaggle: "SMS Spam
   Collection Dataset").
2. Replace dataset/spam.csv, keeping two columns: label (spam/ham) and
   message.
"""

import random
import pandas as pd

random.seed(42)

spam_templates = [
    "Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now: {link}",
    "URGENT! Your account has been suspended. Verify your details immediately at {link}",
    "You have been selected for a FREE cruise to the Bahamas! Call {phone} now to claim.",
    "WINNER!! As a valued customer you have been selected to receive a prize. Reply YES to claim.",
    "Get a loan approved in minutes with NO credit check! Apply now: {link}",
    "Your mobile number has won £2000 in our weekly draw! To claim call {phone}",
    "FREE entry into our £250 weekly competition just text WIN to 80086 now!",
    "Hot singles in your area are waiting to chat with you tonight! Click {link}",
    "You have 1 new voicemail. To listen dial {phone}, charges may apply.",
    "Limited time offer! Buy 1 get 1 FREE on all products. Shop now at {link}",
    "Your bank account has unusual activity. Confirm identity now at {link} or it will be locked.",
    "Claim your FREE iPhone 15 today! Only a few left, click {link} before it's gone!",
    "Reply STOP to unsubscribe or continue to receive exclusive deals and cash prizes weekly.",
    "Dear customer, you are eligible for a tax refund of $850. Submit your details at {link}",
    "Act now! Your prepaid balance has expired, recharge instantly with 50% bonus at {link}",
]

ham_templates = [
    "Hey, are we still on for lunch tomorrow at noon?",
    "Can you pick up some milk on your way home?",
    "Happy birthday! Hope you have an amazing day, let's celebrate this weekend.",
    "Meeting got moved to 3pm, see you in the conference room.",
    "Thanks for helping me move last weekend, really appreciate it!",
    "I'm running about 10 minutes late, traffic is bad on the highway.",
    "Don't forget mom's doctor appointment is at 4pm today.",
    "Did you finish the assignment? I still need to review chapter 5.",
    "The movie was great, we should watch the sequel next week.",
    "Can you send me the notes from today's class? I missed the last part.",
    "Just landed, will call you once I get to the hotel.",
    "Let's grab coffee this Friday, I have a lot to catch you up on.",
    "The package should arrive by Thursday according to the tracking info.",
    "Reminder: gym session at 6am tomorrow, don't oversleep!",
    "I left the keys under the mat, let yourself in whenever you get here.",
]

links = ["bit.ly/claim-now", "tinyurl.com/prize2026", "freegift-offer.net", "win-cash.co", "clickhere-now.info"]
phones = ["09061234567", "07700900123", "08001234567", "07911123456"]

rows = []
for i in range(250):
    t = random.choice(spam_templates)
    msg = t.format(link=random.choice(links), phone=random.choice(phones)) if ("{link}" in t or "{phone}" in t) else t
    rows.append({"label": "spam", "message": msg})

for i in range(500):
    msg = random.choice(ham_templates)
    rows.append({"label": "ham", "message": msg})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/CODSOFT_ML_Internship/Task4_Spam_SMS_Detection/dataset/spam.csv", index=False)
print(df["label"].value_counts())
print(f"Saved {len(df)} rows to dataset/spam.csv")
