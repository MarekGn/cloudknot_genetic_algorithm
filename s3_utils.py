import boto3
import time
import numpy as np

def wait_for_other_workers(workers_num):
    client = boto3.client('s3')
    for i in range(60):
        results = client.list_objects(Bucket='besttictactoe')
        if len(results["Contents"]) == workers_num:
            return True
        elif i == 59:
            return False
        time.sleep(1)


def init_times():
    times = {}
    client = boto3.client('s3')
    results = client.list_objects(Bucket='besttictactoe')
    for file_dic in results["Contents"]:
        print(file_dic)
        print(type(file_dic["Key"]))
        print(type(file_dic["LastModified"]))

        times[file_dic["Key"]] = file_dic["LastModified"]
    return times


def check_modifies(times):
    client = boto3.client('s3')
    results = client.list_objects(Bucket='besttictactoe')
    for i in range(10):
        for file_dic in results["Contents"]:
            if times[file_dic["Key"]] != file_dic["LastModified"]:
                times[file_dic["Key"]] = file_dic["LastModified"]
            elif i == 9:
                return False
            else:
                time.sleep(1)
        return True


def download_pops():
    pop = []
    client = boto3.client('s3')
    results = client.list_objects(Bucket='besttictactoe')
    for i in results["Contents"]:
        client.download_file('besttictactoe', i['Key'], i['Key'])
    for i in results["Contents"]:
        for j in np.load(i['Key']):
            pop.append(j)
    return pop

