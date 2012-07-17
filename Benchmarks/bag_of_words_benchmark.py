#!/usr/bin/env python2.7

import competition_io as cio
import features
import re
from sklearn.ensemble import RandomForestRegressor

def main():    
    predictions = {}

    for essay_set in cio.get_essay_sets():
        print("Making Predictions for Essay Set %s" % essay_set)
        train = list(cio.essays_by_set(essay_set))
        bag = features.train_bag(" ".join(x["EssayText"] for x in train), 500)
        fea =  [features.bag_representation(bag, x["EssayText"]) for x in train]
        rf = RandomForestRegressor(n_estimators = 50)
        rf.fit(fea,[float(x["Score1"]) for x in train])

        test = list(cio.essays_by_set(essay_set, "../Data/public_leaderboard_rel_2.tsv"))
        fea = [features.bag_representation(bag, x["EssayText"]) for x in test]
        predicted_scores = rf.predict(fea)
        for essay_id, pred_score in zip([x["Id"] for x in test], predicted_scores):
            predictions[essay_id] = round(pred_score)
    
    output_file = "../Submissions/bag_of_words_benchmark.csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("id,essay_score\n")
    for key in sorted(predictions.keys()):
        f.write("%d,%d\n" % (key,predictions[key]))
    f.close()
    
if __name__=="__main__":
    main()
