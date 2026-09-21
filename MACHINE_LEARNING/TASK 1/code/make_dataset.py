"""
make_dataset.py
----------------
Generates a sample movie (plot_summary, genre) dataset for Task 1.

NOTE: The official CodSoft dataset link points to a Kaggle dataset
(Genre Classification Dataset - IMDb) that requires a Kaggle login to
download and cannot be fetched automatically in this environment.
This script builds a smaller, representative sample dataset with the
SAME COLUMN FORMAT (title, genre, plot) so the full pipeline in
train_model.py can be run and reproduced end-to-end.

TO USE THE REAL DATASET INSTEAD:
1. Download it from the CodSoft task link (Kaggle: "Genre Classification
   Dataset IMDB").
2. Replace dataset/movies.csv with the downloaded train_data.txt /
   train_data.csv, keeping columns: title, genre, plot.
3. Re-run code/train_model.py — no other changes needed.
"""

import pandas as pd
import random

random.seed(42)

# A bank of plot-summary style sentences per genre, combined to build
# many synthetic-but-realistic plots.
genre_templates = {
    "action": [
        "A former special forces operative is pulled back into action when his family is kidnapped by a ruthless arms dealer.",
        "A rogue cop must stop a terrorist plot to bomb the city's subway system before rush hour.",
        "An elite team of mercenaries is hired to take down a drug cartel controlling a border town.",
        "A stunt driver becomes entangled in a high-speed heist gone wrong across three countries.",
        "A retired assassin is forced out of hiding after a bounty is placed on his head.",
        "Soldiers fight to defend a besieged outpost against overwhelming enemy forces.",
        "A bodyguard races against time to protect a witness from a hitman sent to silence her.",
    ],
    "comedy": [
        "Two mismatched roommates accidentally start a food truck business that spirals into chaos.",
        "A disorganized wedding planner must save a disastrous celebrity wedding in 24 hours.",
        "A man pretends to be his twin brother to win back his ex-girlfriend, with hilarious results.",
        "A group of friends attempt a cross-country road trip that keeps going comically wrong.",
        "An awkward intern is mistaken for the new CEO on his first day at a chaotic startup.",
        "A family reunion turns into a series of embarrassing misunderstandings and pranks.",
        "A struggling actor takes a job as a clown for children's parties to pay rent.",
    ],
    "drama": [
        "A young woman confronts her estranged father after learning she has only months to live.",
        "A struggling single mother fights to keep custody of her children amid financial hardship.",
        "Two brothers reconnect after decades apart to settle their late father's estate.",
        "A teacher in an underfunded school works to inspire a group of at-risk students.",
        "A war veteran struggles to readjust to civilian life after returning home.",
        "A daughter cares for her mother battling a degenerative illness while pursuing her dreams.",
        "An immigrant family navigates cultural identity and hardship in a new country.",
    ],
    "horror": [
        "A family moves into an old farmhouse, unaware of the malevolent spirit that resides within.",
        "A group of campers is stalked by an unseen creature deep in the woods.",
        "A cursed video tape brings death to anyone who watches it within seven days.",
        "A young couple discovers their new home was built on the site of a horrific massacre.",
        "Strange occurrences plague a hospital after a patient dies during a failed experiment.",
        "A babysitter is trapped in a house with something that isn't human.",
        "An archaeologist unleashes an ancient evil after opening a sealed tomb.",
    ],
    "romance": [
        "Two strangers fall in love during a chance encounter on a transatlantic flight.",
        "A florist and a chef reconnect years after a summer romance that never got closure.",
        "A workaholic executive falls for the free-spirited artist renovating her office building.",
        "Childhood friends realize their feelings for each other the night before one of them moves away.",
        "A widow finds unexpected love with her late husband's best friend.",
        "A journalist falls for the subject of her exposé, complicating her assignment.",
        "Two rival bakery owners fall in love while competing in a city-wide baking contest.",
    ],
    "sci-fi": [
        "A crew of astronauts discovers a signal from a planet that shouldn't have life.",
        "In a dystopian future, a hacker uncovers a conspiracy controlling humanity's memories.",
        "A scientist's time-travel experiment threatens to unravel the fabric of reality.",
        "Humanity's last colony must fend off an alien invasion using salvaged technology.",
        "An AI designed to help humanity begins making decisions no one can predict or control.",
        "A group of researchers on a distant space station battle a rapidly evolving organism.",
        "A courier smuggles a mysterious artifact across a war-torn, radiation-scarred Earth.",
    ],
    "thriller": [
        "A detective races to catch a serial killer who leaves cryptic clues at every crime scene.",
        "A woman wakes up with no memory of the last three days, only a body in her trunk.",
        "A journalist uncovers a government cover-up that puts her life in danger.",
        "A hostage negotiator must outwit a captor with a hidden agenda inside a besieged bank.",
        "A programmer discovers his new app is being used to track and eliminate political dissidents.",
        "A stranger's arrival exposes long-buried secrets in a quiet suburban town.",
        "An insurance investigator suspects a client's death was staged for a massive payout.",
    ],
    "documentary": [
        "Filmmakers follow a community rebuilding after a devastating natural disaster.",
        "An in-depth look at the rise and fall of a once-dominant tech company.",
        "Archival footage and interviews trace the history of a decades-long civil rights movement.",
        "A year in the life of wildlife rangers protecting an endangered species from poachers.",
        "Scientists document the effects of climate change on a remote arctic ecosystem.",
        "The film explores the untold story of workers who built a nation's railways.",
        "A portrait of musicians keeping a fading regional folk tradition alive.",
    ],
    "animation": [
        "A young dragon must prove herself to her clan by embarking on a perilous quest.",
        "Toys in a forgotten attic come to life to find their way back to their owner.",
        "A shy robot ventures beyond its factory to discover the world outside.",
        "Animal friends in an enchanted forest band together to stop a greedy developer.",
        "A gifted young wizard learns to control her powers with the help of a talking cat.",
        "Two rival kingdoms of insects must unite against a common predator.",
        "A boy discovers a portal to a magical realm hidden inside his grandmother's clock.",
    ],
    "family": [
        "A city family adapts to life on a rural farm after inheriting it from a distant relative.",
        "A lonely boy befriends a stray dog who helps him make friends at his new school.",
        "Siblings work together to save their grandfather's failing toy shop before Christmas.",
        "A young girl trains her injured falcon to fly again with her father's help.",
        "A father and daughter reconnect on a summer road trip to visit national parks.",
        "Kids at a summer camp band together to solve the mystery of a hidden treasure.",
        "A family adopts a rescue horse that teaches them the meaning of patience and trust.",
    ],
}

rows = []
movie_id = 1
for genre, sentences in genre_templates.items():
    # Combine sentences pairwise/pairs to create more varied "plots" (about 40 per genre)
    for i in range(40):
        s1 = random.choice(sentences)
        s2 = random.choice(sentences)
        plot = s1 if s1 == s2 else f"{s1} {s2}"
        title = f"{genre.title()} Movie {i+1}"
        rows.append({"id": movie_id, "title": title, "genre": genre, "plot": plot})
        movie_id += 1

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/CODSOFT_ML_Internship/Task1_Movie_Genre_Classification/dataset/movies.csv", index=False)
print(df["genre"].value_counts())
print("Saved", len(df), "rows to dataset/movies.csv")
