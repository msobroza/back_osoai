import base64
import json
import boto3
import numpy as np
from scipy.io import wavfile
import six
import csv
import vggish_input
import vggish_params
import vggish_postprocess
import csv
import sys
import math
import boto3
import io
from datetime import datetime, timedelta
import time

import scipy as scp

#import resampy

print('Loading function')

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128

def get_dict_sounds():
    index_sound = dict()
    index_threshold = dict()
    with open('class_labels_indices.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        c = 0
        for row in spamreader:
            c += 1
            if c == 1:
                continue
            index_sound[int(row[0])] = str(row[5])
            index_threshold[int(row[0])] = float(row[7])
    return index_sound, index_threshold

index_sound, index_threshold = get_dict_sounds()
# Prepare a postprocessor to munge the model embeddings.
pproc = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
    
def process_records_solution2(events):
    

    sound_array = list()
    #Kinesis data is base64 encoded so decode here
    print(events)
    payload=base64.b64decode(events['Records'][0]["kinesis"]["data"])
    sound_array.append(np.frombuffer(payload, dtype=np.int16).reshape(-1))

   
    #------------------------------------------------------
    # Save received sound to wave file and upload it to S3
    #------------------------------------------------------
    # scp.io.wavfile.write('/tmp/output.wav', 16000, np.frombuffer(payload, dtype=np.int16))
    # s3 = boto3.client('s3')
    # filename = '/tmp/output.wav'
    # bucket_name = 'micro-sound'
    # s3.upload_file(filename, bucket_name, filename)

    sound_array = np.concatenate(sound_array, axis=None).reshape(-1)

    examples_batch = vggish_input.array_to_examples(sound_array)

    client = boto3.client('runtime.sagemaker', region_name='eu-west-1')
    print(examples_batch.shape)

    data = np.expand_dims(examples_batch, axis=-1).tolist()

    endpoint_feat_extract = 'sagemaker-tensorflow-2019-02-11-15-08-36-462' 
    endpoint_classifier = 'sagemaker-tensorflow-2019-02-11-15-15-34-851'
    response = client.invoke_endpoint(EndpointName=endpoint_feat_extract, Body=json.dumps(data))
 
    body = response['Body'].read().decode('utf-8')

    embedding_sound = np.array(json.loads(body)['outputs']['vgg_features']['floatVal']).reshape(-1, vggish_params.EMBEDDING_SIZE)
    num_secs = 10

    postprocessed_batch_keras = pproc.postprocess_single_sample(embedding_sound, num_secs)
    postprocessed_batch_keras = uint8_to_float32(postprocessed_batch_keras)

    input_class = np.swapaxes(np.swapaxes(postprocessed_batch_keras, 0, 2), 1, 2).tolist()
 
    response = client.invoke_endpoint(EndpointName=endpoint_classifier, Body=json.dumps(input_class))

    body = response['Body'].read().decode('utf-8')
  
    output = np.array(json.loads(body)['outputs']['output']['floatVal']).reshape(-1, len(index_sound))
  
    if len(output.shape) == 2:
        output = np.mean(output, axis=0)
  
    indexes_max = output.argsort()[-10:][::-1]

    min_threshold = 0.05
    results={}
    for i in indexes_max:
        print(index_sound[i])
        print(output[i])
        if output[i] > min_threshold:
            if index_threshold[i] < output[i]:
                results[index_sound[i]]= output[i]
                
    data_to_send ={}
    data_to_send['results']=results


    try:
   
    
        
        invokeLam = boto3.client("lambda", region_name="eu-west-1")
        function_to_call = 'arn:aws:lambda:eu-west-1:917211837119:function:OsoPoc_SendResultsToUser_WebSocket'
        resp = invokeLam.invoke(FunctionName = function_to_call, InvocationType = "Event", Payload = json.dumps(data_to_send))
    
    
    except :
        print('invoke lambda send error')
        return 0

    
        
   

    return 0


def lambda_handler(event, context):
   
     
    
    oso_poc_stream_name = 'abc'  #'OsoPoc-DataStream'

    kinesis_client = boto3.client('kinesis', region_name='eu-west-1')
    
    
    #print(event['Records'][0]['kinesis'])
    
    
    #------------------------------
    # Solution 2 : smartphone side
    #------------------------------        
    
    process_records_solution2(event)    
    

    
    
    
    # arrivalTimestamp_datetime = datetime.utcfromtimestamp(arrivalTimestamp)


    # delay = datetime.utcnow() - arrivalTimestamp_datetime    
    # print('delay', delay)
    # print('arrival time',  arrivalTimestamp_datetime)
    # startTime = arrivalTimestamp_datetime - timedelta(seconds=15) 
    # print('startTime',  startTime)
    
    
    
        
    return 'Successfully processed {} records.'.format(len(event['Records']))
