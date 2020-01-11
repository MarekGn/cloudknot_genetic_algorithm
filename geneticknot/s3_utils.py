import boto3
import time
import numpy as np


def wait_for_other_workers(workers_num, bucket_name):
    client = boto3.client('s3')
    for i in range(300):
        results = client.list_objects(Bucket=bucket_name)
        if len(results["Contents"]) == workers_num:
            return True
        elif i == 299:
            return False
        time.sleep(1)


def init_times(bucket_name):
    times = {}
    client = boto3.client('s3')
    results = client.list_objects(Bucket=bucket_name)
    for file_dic in results["Contents"]:
        times[file_dic["Key"]] = file_dic["LastModified"]
    return times


def check_modifies(times, bucket_name, workers_num):
    client = boto3.client('s3')
    for i in range(300):
        results = client.list_objects(Bucket=bucket_name)
        check = 0
        for file_dic in results["Contents"]:
            if times[file_dic["Key"]] != file_dic["LastModified"]:
                check += 1
        if check == workers_num:
            for file_dic in results["Contents"]:
                times[file_dic["Key"]] = file_dic["LastModified"]
            return True
        elif i == 299:
            return False
        time.sleep(1)


def download_pops(bucket_name):
    pop = []
    client = boto3.client('s3')
    results = client.list_objects(Bucket=bucket_name)
    for i in results["Contents"]:
        client.download_file(bucket_name, i['Key'], i['Key'])
    for i in results["Contents"]:
        for j in np.load(i['Key'], allow_pickle=True):
            pop.append(j)
    return pop

