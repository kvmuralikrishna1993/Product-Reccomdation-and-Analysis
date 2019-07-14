import csv
import json

csvfile = open('sentiment_test.csv', 'r', encoding = "ISO-8859-1")
reader = csv.DictReader(csvfile)
output =[]
for each in reader:
    row ={}
    row['text'] = each['tweet']
    row['label']  = each['polarity']
    output.append(row)
json.dump(output,open('sentiment_test.json','w'),indent=4,sort_keys=False)