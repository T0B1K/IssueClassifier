import pika

def process_issues(request_issues):
    rmq_connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    rmq_channel = rmq_connection.channel()
    rmq_channel.exchange_declare(
        exchange='classify_issue', exchange_type='fanout')
    
    try:
        count = 0
        failure = False
        for issue in request_issues:
            rmq_channel.basic_publish(
            exchange='classify_issue', routing_key='', body=issue)
            count += 1
    except:
        failure = True
    finally:
        rmq_connection.close()
        if failure:
            return ({"Failure": "Error occured while submitting issue(s) for classification."}, 500)
        else:
            return ({"Success": "Submitted " + count * " issue(s) for classification"})
