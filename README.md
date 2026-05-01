# Reinforcement Learning – FrozenLake (Q-Learning)

This project demonstrates Q-Learning on the FrozenLake environment using Python and Gymnasium. It includes a Flask web app to visualize learning performance and final policy.

## 🚀 Features
- Q-Learning algorithm (from scratch)
- Epsilon-greedy exploration strategy
- Reward-per-episode graph
- Final policy visualization (arrow grid)
- Flask-based web interface

## 🧠 Concepts Covered
- States, Actions, Rewards
- Q-table
- Exploration vs Exploitation
- Reinforcement Learning basics

## 🛠 Tech Stack
- Python
- Gymnasium
- NumPy
- Matplotlib
- Flask

## 📁 Project Structure
rl_frozenlake_project/
│
├── app.py
├── train.py
├── templates/
│ └── index.html
├── static/
│ └── style.css
├── requirements.txt
└── README.md


## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
python app.py

Open in browser:

http://127.0.0.1:5000/

📊 Output
Reward graph showing learning over episodes
Final policy displayed as arrow grid
Success rate of last 100 episodes

🎯 Environment
FrozenLake-v1 (stochastic environment)

📌 Author
Shivam Korgaonkar
