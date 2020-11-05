import base64
import os
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
from predict_video import predict_video
from recognition import E2E

IP = os.environ.get("IP", "127.0.0.1")
PORT = os.environ.get("PORT", "8888")
# UPLOAD_DIRECTORY = "/Users/lephuocmy/Desktop/linhtinh/React-Landing-Page-Template/face-recognition-dash/static/app_uploaded_files"
UPLOAD_DIRECTORY= os.getcwd() + "/app_uploaded_files"
STATIC_VIDEO_URL = "http://{}:{}/download/".format(IP, PORT)

print("====os.getcwd()======", os.path.abspath(os.getcwd()))
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)


@server.route("/download/<path:path>")
def download(path):
    print("============ ", path)
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

app.layout = html.Div(
    [
        html.Div([
            html.H1("License plate recognition demo"),
        ]),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Drag and drop or click to select a video to upload."]
            ),
            style={
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
            },
            multiple=True,
        ),
        html.Div(id="videos"),
    ],
)


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    print("=========", UPLOAD_DIRECTORY)

    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

    input_path = "/src/app_uploaded_files/{}".format(name)
    output_path = "/src/app_uploaded_files/output_{}".format(name[:-4] + ".webm")
    text_path = "/src/app_uploaded_files/text_result.txt"

    print("file {} exists: {}".format(input_path, os.path.exists(input_path)))
    
    predict_video(input_path, output_path, text_path)

def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "{}/{}".format(STATIC_VIDEO_URL, filename)
    return html.Video([
        html.Source(src=location, type="video/webm"),
        html.Source(src=location, type="video/mov"),
        html.Source(src=location, type="video/mp4")
    ], controls='controls', width="100%"),

@app.callback(
    Output("videos", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""
    file_name = ""
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            ts = datetime.datetime.now().timestamp()
            file_name = str(ts) + name
            save_file(file_name, data)

    if file_name == "":
        return None

    return html.Div(
        [
            html.Div(file_download_link(file_name), id="input-video", className="six columns x-container"),
            html.Div(file_download_link("output_{}".format(file_name[:-4] + ".webm")), id="output-video", className="six columns x-container"),
        ],
        className="box",
    )


@server.route('/static/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return send_from_directory(os.path.join(root_dir, 'static'), path)

if __name__ == "__main__":
    app.run_server(debug=True, port=PORT)