from prometheus_client import Gauge

time_metric = Gauge('time_metric', 'Sum of the kernel execution times')
