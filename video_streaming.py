import ffmpeg_streaming
from ffmpeg_streaming import S3, CloudManager, Formats, Representation, Size, Bitrate

ACCESS_KEY = "AKIARXHPRX3OPTDR6Z6M"
SECRET_KEY = "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"
REGION_NAME = "ap-southeast-1"

BUCKET_NAME = "lephuocmy686868"
TEST_VIDEO = "test.MOV"

# see https://docs.aws.amazon.com/general/latest/gr/aws-security-credentials.html to get Security Credentials
s3 = S3(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION_NAME)

# save_to_s3 = CloudManager().add(s3, bucket_name=BUCKET_NAME, folder="folder/subfolder")
# A filename can also be passed to the CloudManager class to change the filename(e.x. CloudManager(filename="hls.m3u8"))

video = ffmpeg_streaming.input(s3, bucket_name=BUCKET_NAME, key=TEST_VIDEO)

# dash = video.dash(Formats.h264())
# dash.auto_generate_representations()

# dash.output()
