from transformers import pipeline

PROMPTS = [
    "Steph Curry is an elite [MASK] shooter.",
    "The Knicks run a lot of pick and [MASK] sets.",
    "Wembanyama protects the [MASK] at an elite level.",
    "The Warriors spam [MASK] actions to create open threes.",
    "The Celtics switch a ton and rely on [MASK] help.",
    "Luka manipulates the defense with [MASK] pacing.",
    "SGA lives in the midrange and gets to the [MASK] line.",
    "The Heat love running [MASK] handoffs.",
    "Dame Time means late-game [MASK] shotmaking.",
    "Jokic is a passing [MASK] who reads the floor.",
]

LARRYBERT1_DIR = "C:/Users/ryous/Downloads/larrybert/models/larrybert-base-mlm/checkpoint-1500"

def run(model_name, name, topk):
    fill = pipeline("fill-mask", model=model_name, tokenizer=model_name, device=-1)
    print(f"\n=== MODEL: {name} ===")
    for p in PROMPTS:
        out = fill(p, top_k=topk)
        preds = ", ".join([f"{x['token_str'].strip()} ({x['score']:.2f})" for x in out])
        print(f"\n{p}\n  -> {preds}")

def main():
    run('bert-base-uncased','bert-base-uncased', 5)
    run(LARRYBERT1_DIR, 'larry-bert-base-mlm', 5)
    print("\n")

main()