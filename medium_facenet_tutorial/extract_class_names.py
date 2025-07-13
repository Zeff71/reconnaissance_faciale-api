# extract_class_names.py
import pickle
import json

with open("C:/Users/Zeff/reconnaissance_faciale/app/output/first_classifier.pkl", "rb") as f:
    _, class_names = pickle.load(f)

with open("C:/Users/Zeff/reconnaissance_faciale/app/output/class_names.json", "w") as f:
    json.dump(class_names, f)

print("✅ class_names.json généré avec succès.")
