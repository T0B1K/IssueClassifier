with open("enhancement.json") as file:
    data = json.loads(file.read())
documents:list = list(map(lambda entry: entry["text"], data))[:4000] 
logging.warning("--------> documents.size: {} ---------".format(len(documents)))
array = numpy.array(documents)
i:int = 0
while i*100 < len(array):
    idx = i*100
    sendingArray = array[idx : idx+100]
    logging.warn("{}: sending: {}".format(idx,len(sendingArray)))
    serialized = pickle.dumps(sendingArray)
    channel.basic_publish(exchange='', routing_key='vecin', body=serialized)
    i=i+1