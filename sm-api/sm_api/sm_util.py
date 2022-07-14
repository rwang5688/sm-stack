import boto3
import json


def invoke_endpoint(endpoint_name, message):
    '''Call the sagemaker (serverless) inference endpoint
    '''
    print("invoke_endpoint: endpoint_name: %s" % (endpoint_name))
    print("invoke_endpoint: message: %s" % (message))

    client = boto3.client('sagemaker-runtime')
    
    content_type = "application/json"
    # must specify "Inputs"
    data = {
        "inputs": message
    }

    ie_response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=json.dumps(data)
    )
    print("invoke_endpoint: ie_response: %s" % (ie_response))

    # Body is a byte stream that needs to be read and decoded
    response = ie_response["Body"].read().decode("utf-8")
    print("invoke_endpoint: response: %s" %(response))

    return(response)

