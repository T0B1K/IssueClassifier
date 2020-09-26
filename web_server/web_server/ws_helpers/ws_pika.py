from typing import List, Tuple

from pika import BlockingConnection, ConnectionParameters

from web_server.ws_helpers.ws_models import UnclassifiedIssue


def _init_pika() -> Tuple[BlockingConnection, BlockingConnection.channel]:
    rmq_connection = BlockingConnection(ConnectionParameters(host='localhost'))
    
    rmq_channel = rmq_connection.channel()
    
    rmq_channel.exchange_declare(
        exchange='classify_issue',
        exchange_type='fanout',
        durable=True)

    return rmq_connection, rmq_channel


def classify_issues(unclassified_issues: List[UnclassifiedIssue]) -> None:
    rmq_connection, rmq_channel = _init_pika()

    try:
        failure = False
        for issue in unclassified_issues:
            rmq_channel.basic_publish(
                exchange='classify_issue', routing_key='', body=issue.json())
    except:
        failure = True
    finally:
        rmq_connection.close()
        if failure:
            print(
                "Failure: Error occured while submitting issue(s) for classification.")
        else:
            print(
                "Success: Submitted issue(s) for classification")
