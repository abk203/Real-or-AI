import keras
import numpy as np
from PIL import Image
import os
import gradio as gr
import random

PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PATH, 'model')
IMAGES = os.path.join(PATH, 'images')

ROUNDS = 10
THRESHOLD = 0.5
IMG_SIZE = 128

PIXEL = 255.0
NAME = {0: "AI-generated", 1: "Real photo"}

LABELS = {}
for (folder, label) in (("real", 1), ("AI", 0)):
    n = os.path.join(IMAGES, folder)
    for fname in sorted(os.listdir(n)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            LABELS[os.path.join(n, fname)] = label
FILES = sorted(LABELS)

model = keras.saving.load_model("best_model.keras", compile=False)
batch = np.stack([np.asarray(Image.open(f).convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0 * PIXEL for f in FILES])

P_Real = dict(zip(FILES, model.predict(batch).ravel().tolist()))

def scoreboard(s):
    played = s["idx"] + s["answered"]
    return f"You: {s["human"]} / {played} | Model: {s["model"]} / {played}"

def show_round(s):
    return (s, s["order"][s["idx"]], f"### Round {s['idx'] + 1} of {len(s['order'])}",
            scoreboard(s), "Real photo or AI-generated? Make your call.",
            gr.update(visible=False),                                  # model confidence
            gr.update(interactive=True), gr.update(interactive=True),  # guess buttons
            gr.update(visible=False), gr.update(visible=False))        # next / play again

def new_game():
    s = {"order": random.sample(FILES, min(ROUNDS, len(FILES))),
         "idx": 0, "human": 0, "model": 0, "answered": False}
    return show_round(s)


def guess(choice, s):
    f = s["order"][s["idx"]]
    truth, p = LABELS[f], P_REAL[f]
    model_guess = int(p >= 0.5)
    s["human"] += choice == truth
    s["model"] += model_guess == truth
    s["answered"] = True

    if choice == truth:
        gr.Info("✅ Correct!")
    else:
        gr.Warning(f"Wrong, it was {NAME[truth]}.")

    feedback = (f"**Answer:** {NAME[truth]}\n\n"
                f"You said {NAME[choice]} {'✅' if choice == truth else '❌'} · "
                f"Model said {NAME[model_guess]} {'✅' if model_guess == truth else '❌'}")

    last = s["idx"] + 1 == len(s["order"])

    if last:
        h, m = s["human"], s["model"]
        result = "You win!" if h > m else "🤖 The model wins!" if m > h else "It's a tie!"
        feedback += f"\n\n## {result}\nFinal score: you {h}, model {m} (out of {len(s['order'])})."

    return (s, gr.update(), f"### Round {s['idx'] + 1} of {len(s['order'])}", scoreboard(s), feedback,
            gr.update(value={"Real photo": p, "AI-generated": 1 - p}, visible=True),
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(visible=not last), gr.update(visible=last))

def next_round(s):
    s["idx"] += 1
    s["answered"] = False
    return show_round(s)

with gr.Blocks(theme=gr.themes.Soft(), title="Real or AI?") as demo:
    gr.Markdown(f"# Real or AI? Can you beat the model?\n"
                f"Guess whether each face is a real photo or AI-generated. Best score after {ROUNDS} rounds wins.")

    state = gr.State()

    with gr.Row():
        with gr.Column():
            face = gr.Image(show_label=False, interactive=False, height=400)

            with gr.Row():
                btn_real = gr.Button("📷 Real photo", variant="primary")
                btn_ai = gr.Button("🤖 AI-generated", variant="primary")

        with gr.Column():
            round_md = gr.Markdown()
            score_md = gr.Markdown()
            feedback_md = gr.Markdown()

            model_conf = gr.Label(label="Model's confidence", visible=False)
            btn_next = gr.Button("Next face", visible=False)
            btn_restart = gr.Button("Play again", visible=False)

    outputs = [state, face, round_md, score_md, feedback_md, model_conf,
               btn_real, btn_ai, btn_next, btn_restart]

    btn_real.click(lambda s: guess(1, s), state, outputs)
    btn_ai.click(lambda s: guess(0, s), state, outputs)
    btn_next.click(next_round, state, outputs)
    btn_restart.click(new_game, None, outputs)

    demo.load(new_game, None, outputs)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), allowed_paths=[IMAGES])


