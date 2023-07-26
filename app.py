import torch
import numpy as np
import supervision as sv
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
colors = sv.ColorPalette.default()
polygons = [
    np.array([
        [0, 0],
        [400 - 5, 0],
        [400 - 5, 200 - 5],
        [0, 200 - 5]
    ], np.int32),
    np.array([
        [0 , 200],
        [200, 200],
        [200, 400 - 5],
        [0, 400 - 5]
    ], np.int32),
    np.array([
        [200,200],
        [300,200],
        [300,400],
        [200,400]
    ],np.int32),
    np.array([
        [300,200],
        [400,200],
        [400,400],
        [300,400]
    ],np.int32),
    np.array([
        [0, 400 + 5],
        [240 - 5, 400 + 5],
        [240 - 5, 600],
        [0, 600]
    ], np.int32),
    np.array([
        [240,400+5],
        [400,400+5],
        [400+5,600+5],
        [240,600+5]
    ],np.int32),
    np.array([
        [0, 600 + 5],
        [400, 600 + 5],
        [400, 750],
        [0 , 750]
    ], np.int32),
    np.array([
        [0,750+5],
        [400+5,750+5],
        [400+5,1000+5],
        [0,1000+5],
    ],np.int32),
]
video_info = sv.VideoInfo.from_video_path("aa.jpeg")

zones = [
    sv.PolygonZone(
        polygon=polygon,
        frame_resolution_wh=video_info.resolution_wh
    )
    for polygon
    in polygons
]
zone_annotators = [
    sv.PolygonZoneAnnotator(
        zone=zone,
        color=colors.by_idx(index),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )
    for index, zone
    in enumerate(zones)
]
box_annotators = [
    sv.BoxAnnotator(
        color=colors.by_idx(index),
        thickness=1,
        text_thickness=1,
        text_scale=0.5
        )
    for index
    in range(len(polygons))
]


# Helper function to annotate an image
def annotate_image(image_data):
    image = np.array(Image.open(io.BytesIO(image_data)))

    # Detect objects in the image
    results = model(image, size=1280)
    detections = sv.Detections.from_yolov5(results)
    detections = detections[(detections.confidence > 0.1)]

    # Annotate the image with detected objects
    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        image = box_annotator.annotate(scene=image, detections=detections_filtered)
        image = zone_annotator.annotate(scene=image)

    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/annotate_image', methods=['POST'])
def annotate_single_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request.'}), 400

    image_file = request.files['image'].read()
    annotated_image = annotate_image(image_file)
    
    # Convert annotated image to a byte stream
    image_byte_array = io.BytesIO()
    annotated_image_pil = Image.fromarray(annotated_image)
    annotated_image_pil.save(image_byte_array, format='JPEG')
    image_byte_array.seek(0)
    
    return image_byte_array.getvalue()

if __name__ == '__main__':
    app.run(debug=True)
