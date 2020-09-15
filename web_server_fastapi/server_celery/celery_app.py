from celery import Celery

app = Celery('celery', backend='redis://localhost:6379/0',
                    broker='pyamqp://guest@localhost//', include=["tasks"])
