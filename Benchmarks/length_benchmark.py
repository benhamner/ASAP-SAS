#!/usr/bin/env python2.7

import competition_io as cio
import re
from sklearn.ensemble import RandomForestRegressor

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
        features = extract_features([x["EssayText"] for x in train], feature_functions)
        rf = RandomForestRegressor(n_estimators = 50)
        rf.fit(features,[float(x["Score1"]) for x in train])

        test = list(cio.essays_by_set(essay_set, "../Data/public_leaderboard_rel_2.tsv"))
        features = extract_features([x["EssayText"] for x in test], feature_functions)
        predicted_scores = rf.predict(features)
        for essay_id, pred_score in zip([x["Id"] for x in test], predicted_scores):
            predictions[essay_id] = round(pred_score)
    
    output_file = "../Submissions/length_benchmark.csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("id,essay_score\n")
    for key in sorted(predictions.keys()):
        f.write("%d,%d\n" % (key,predictions[key]))
    f.close()
    
if __name__=="__main__":
    main()
