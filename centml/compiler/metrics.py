from prometheus_client import Gauge, Counter

time_metric = Gauge('time_metric', 'Sum of the kernel execution times')