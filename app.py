from ultralytics import YOLO
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# Load a model
model = YOLO("best.pt")


def inference(gr_input):
    """
    Inference function for gradio.
    """
    pred = model(gr_input)
    draw_prediction = ImageDraw.Draw(gr_input)
    boxes_predict = pred[0].boxes
    boxes = boxes_predict.xyxy.tolist()
    scores = boxes_predict.conf.tolist()
    for score, box in zip(scores, boxes):
        x, y, x2, y2 = tuple(box)
        draw_prediction.rectangle((x, y, x2, y2), outline="red", width=2)
    return gr_input


def main():
    imagein = gr.inputs.Image(label="Input Image", type="pil")
    imageout = gr.outputs.Image(label="Predicted Image", type="pil")

    interface = gr.Interface(
        fn=inference,
        inputs=imagein,
        outputs=imageout,
        title="Potholes detection",
        interpretation="default",
    ).launch(debug="True", share="True")

    # launch demo
    interface.launch()

if __name__ == "__main__":
    main() 
