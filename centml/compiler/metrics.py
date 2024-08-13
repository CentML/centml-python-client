from prometheus_client import Gauge

A10_time_metric = Gauge('A10_time', 'Sum of the kernel execution times on A10')
