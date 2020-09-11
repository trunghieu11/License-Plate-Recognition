import boto3
import time
import json
import os
import predict_video

# Create SQS client
sqs = boto3.client(
    'sqs', 
    aws_access_key_id="",
    aws_secret_access_key="",
    region_name=""
)
s3 = boto3.client(
    's3',
    aws_access_key_id="",
    aws_secret_access_key="",
    region_name=""
)
bucket = ""
queue_url = ''


while True:
    try:
        time.sleep(1)

        # Receive message from SQS queue
        response = sqs.receive_message(
            QueueUrl=queue_url,
            AttributeNames=[
                'SentTimestamp'
            ],
            MaxNumberOfMessages=1,
            MessageAttributeNames=[
                'All'
            ],
            VisibilityTimeout=0,
            WaitTimeSeconds=0
        )

        if "Messages" in response:
            message = response['Messages'][0]
            receipt_handle = message["ReceiptHandle"]
            body = json.loads(message["Body"])
            ev_ms =  json.loads(body["Message"])
            file = ev_ms["Records"][0]["s3"]["object"]["key"]
            print("Dowloading", file)

            # download file that need to detect
            path = "/Users/trunghieu11/Google Drive/Work/My_Projects/License-Plate-Recognition/{}".format(file)
            print(path)
            print(file)
            print("=" * 50)
            fname = os.path.basename(path)
            print("=========", fname)
            output_path = "result/{}".format(fname)
            s3.download_file(bucket, file , path)
            print("dowloaded")

            # do detection
            predict_video.predict_video(path, output_path)

            # upload result
            print("Uploading detection result: {}".format(fname))
            response = s3.upload_file(output_path, bucket, "result/{}".format(fname))

            # Delete received message from queue
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
            print('Deleted message: %s' % receipt_handle)
        else:
            print("There is no message to consume")
    except Exception as e:
        print("Error had occured while consume message", e)