#!/usr/bin/env python2.7

import competition_io as cio
import features
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def add_essay_training(data, essay_set, essay, score):
    if essay_set not in data:
        data[essay_set] = {"essay":[],"score":[]}
    data[essay_set]["essay"].append(essay)
    data[essay_set]["score"].append(score)

def add_essay_test(data, essay_set, essay, prediction_id):
    if essay_set not in data:
        data[essay_set] = {"essay":[], "prediction_id":[]}
    data[essay_set]["essay"].append(essay)
    data[essay_set]["prediction_id"].append(prediction_id)

def get_character_count(essay):
    return len(essay)

def get_word_count(essay):
    return len(re.findall(r"\s", essay))+1

def extract_features(essays, feature_functions):
    return [[f(es) for f in feature_functions] for es in essays]

def main():    
    feature_functions = [get_character_count, get_word_count]
    predictions = {}

    for essay_set in cio.get_essay_sets():
        print("Making Predictions for Essay Set %s" % essay_set)
        train = list(cio.essays_by_set(essay_set))
        bag = features.train_bag(" ".join(x["EssayText"] for x in train), 1000)
        print(bag)
        fea =  [features.bag_representation(bag, x["EssayText"]) for x in train]
        #features = extract_features([x["EssayText"] for x in train], feature_functions)
        rf = RandomForestClassifier(n_estimators = 100)
        rf.fit(fea,[float(x["Score1"]) for x in train])

        test = list(cio.essays_by_set(essay_set, "../Data/public_leaderboard.tsv"))
        #features = extract_features([x["EssayText"] for x in test], feature_functions)
        fea = [features.bag_representation(bag, x["EssayText"]) for x in test]
        predicted_scores = rf.predict(fea)
        for essay_id, pred_score in zip([x["Id"] for x in test], predicted_scores):
            predictions[essay_id] = round(pred_score)
    
    output_file = "../Submissions/bag_benchmark.csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("id,essay_score\n")
    for key in sorted(predictions.keys()):
        f.write("%d,%d\n" % (key,predictions[key]))
    f.close()
    
if __name__=="__main__":
    main()
