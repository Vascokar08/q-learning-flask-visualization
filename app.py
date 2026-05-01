from flask import Flask, render_template
from train import train_agent
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    Q, rewards, success_rate = train_agent()

    # Graph
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # Policy
    actions = ["←", "↓", "→", "↑"]
    policy = np.argmax(Q, axis=1).reshape(4, 4)

    return render_template(
        "index.html",
        graph=graph_url,
        policy=policy,
        actions=actions,
        success_rate=success_rate
    )

if __name__ == "__main__":
    app.run(debug=True)