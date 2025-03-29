import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
import requests
from time import sleep
import random
import sys
warnings.filterwarnings("ignore")

NUM_ROUNDS = 5
host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

data_str = '''{
    "1": {"text": "Feather", "cost": 1},
    "2": {"text": "Coal", "cost": 1},
    "3": {"text": "Pebble", "cost": 1},
    "4": {"text": "Leaf", "cost": 2},
    "5": {"text": "Paper", "cost": 2},
    "6": {"text": "Rock", "cost": 2},
    "7": {"text": "Water", "cost": 3},
    "8": {"text": "Twig", "cost": 3},
    "9": {"text": "Sword", "cost": 4},
    "10": {"text": "Shield", "cost": 4},
    "11": {"text": "Gun", "cost": 5},
    "12": {"text": "Flame", "cost": 5},
    "13": {"text": "Rope", "cost": 5},
    "14": {"text": "Disease", "cost": 6},
    "15": {"text": "Cure", "cost": 6},
    "16": {"text": "Bacteria", "cost": 6},
    "17": {"text": "Shadow", "cost": 7},
    "18": {"text": "Light", "cost": 7},
    "19": {"text": "Virus", "cost": 7},
    "20": {"text": "Sound", "cost": 8},
    "21": {"text": "Time", "cost": 8},
    "22": {"text": "Fate", "cost": 8},
    "23": {"text": "Earthquake", "cost": 9},
    "24": {"text": "Storm", "cost": 9},
    "25": {"text": "Vaccine", "cost": 9},
    "26": {"text": "Logic", "cost": 10},
    "27": {"text": "Gravity", "cost": 10},
    "28": {"text": "Robots", "cost": 10},
    "29": {"text": "Stone", "cost": 11},
    "30": {"text": "Echo", "cost": 11},
    "31": {"text": "Thunder", "cost": 12},
    "32": {"text": "Karma", "cost": 12},
    "33": {"text": "Wind", "cost": 13},
    "34": {"text": "Ice", "cost": 13},
    "35": {"text": "Sandstorm", "cost": 13},
    "36": {"text": "Laser", "cost": 14},
    "37": {"text": "Magma", "cost": 14},
    "38": {"text": "Peace", "cost": 14},
    "39": {"text": "Explosion", "cost": 15},
    "40": {"text": "War", "cost": 15},
    "41": {"text": "Enlightenment", "cost": 15},
    "42": {"text": "Nuclear Bomb", "cost": 16},
    "43": {"text": "Volcano", "cost": 16},
    "44": {"text": "Whale", "cost": 17},
    "45": {"text": "Earth", "cost": 17},
    "46": {"text": "Moon", "cost": 17},
    "47": {"text": "Star", "cost": 18},
    "48": {"text": "Tsunami", "cost": 18},
    "49": {"text": "Supernova", "cost": 19},
    "50": {"text": "Antimatter", "cost": 19},
    "51": {"text": "Plague", "cost": 20},
    "52": {"text": "Rebirth", "cost": 20},
    "53": {"text": "Tectonic Shift", "cost": 21},
    "54": {"text": "Gamma-Ray Burst", "cost": 22},
    "55": {"text": "Human Spirit", "cost": 23},
    "56": {"text": "Apocalyptic Meteor", "cost": 24},
    "57": {"text": "Earth’s Core", "cost": 25},
    "58": {"text": "Neutron Star", "cost": 26},
    "59": {"text": "Supermassive Black Hole", "cost": 35},
    "60": {"text": "Entropy", "cost": 45}
}'''
dict = json.loads(data_str)


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def get_embedding(text, max_length=32):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0).cpu().numpy()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def rank_candidates(player_b_word, dict):
    target_phrase = f"{player_b_word} crushed"
    target_embedding = get_embedding(target_phrase)
    
    candidate_scores = []
    for num, info in dict.items(): 
        candidate_text = info["text"]
        candidate_embedding = get_embedding(candidate_text)
        score = cosine_similarity(candidate_embedding, target_embedding)
        candidate_scores.append((num, candidate_text, score))
        
    candidate_scores.sort(key=lambda x: x[2], reverse=True)
    return candidate_scores


def what_beats(word):
    ranked_candidates = rank_candidates(word, dict)
    best5 = ranked_candidates[:5]
    best_candidate = min(best5, key=lambda x: int(x[0]))
    return best_candidate[0]

def play_game(player_id):
    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            print(f"Get: {response.json()}")
            sys_word = response.json()['word']
            round_num = response.json()['round']
            sleep(1)
        if round_id > 1:
            status = requests.get(status_url)
            print(f"Round_status: {status.json()}")
        chosen_word = what_beats(sys_word)
        
        data = {"player_id": player_id, "word_id": chosen_word, "round_id": round_id}
        response = requests.post(post_url, json=data)
        print(f"Post: {response.json()}")

if __name__ == "__main__":
    player_id = "5XTm2JDek9"
    play_game(player_id)
