from transformers import pipeline

PROMPTS = [
    "Shai Gilgeous-Alexander is elite at drawing [MASK] on drives to the rim.",
    "Dyson Daniels is a poor [MASK] point shooter but provides elite perimeter defense.",
    "Patty Mills thrives as a [MASK] guard coming off the bench.",
    "Nikola Jokic is one of the best passing [MASK] in NBA history.",
    "Jayson Tatum finished the game with 32 points and 10 [MASK].",
    "Luka Doncic recorded a triple-[MASK] in the Mavericks win.",
    "Anthony Davis dominated the paint with 18 [MASK] tonight.",
    "Stephen Curry knocked down seven [MASK] in the fourth quarter.",
    "The Lakers struggled with defensive [MASK] throughout the first half.",
    "Milwaukee's offensive [MASK] improved after the timeout.",
    "Boston plays at a much faster [MASK] than most teams in the league.",
    "The lineup posted a +15 [MASK] in just eight minutes.",
    "Draymond Green anchors the Warriors' defensive [MASK].",
    "The Suns rely heavily on Devin Booker as their primary [MASK] creator.",
    "Victor Wembanyama protects the [MASK] at an elite level.",
    "The Nuggets stagger Jokic with the second [MASK].",
    "The Warriors run a lot of [MASK] and [MASK] actions for Stephen Curry.",
]

LARRYBERT1_DIR = r"N:\LARRYBERT\LarryBERT\models\larry-bert-base\checkpoint-3000"

def run(model_name, name, topk):
    fill = pipeline("fill-mask", model=model_name, tokenizer=model_name, device=-1)
    print(f"\n=== MODEL: {name} ===")
    for p in PROMPTS:
        out = fill(p, top_k=topk)
        preds = ", ".join([f"{x['token_str'].strip()} ({x['score']:.2f})" for x in out])
        print(f"\n{p}\n  -> {preds}")

def main():
    # run('bert-base-uncased','bert-base-uncased', 5)
    run(LARRYBERT1_DIR, 'larry-bert-base-mlm', 5)
    print("\n")

main()