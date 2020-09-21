import json

def openFile(filename):
    with open(filename, "r",  encoding='utf-8', errors='ignore') as file:
        data=file.read()
    #we just take all the "text" from the JSON
    return list(map(lambda entry: entry["text"], json.loads(data)))

def saveWrongClassifiedToFile(filename, data):
    f = open(filename, "w")
    jsonData = []
    for x in data:
        jsonData.append({"text": x})
        # convert into JSON:
    f.write(json.dumps(jsonData))
    f.close()

tmp = openFile("../documents/api.json")
saveWrongClassifiedToFile("../documents/api1.json", tmp)