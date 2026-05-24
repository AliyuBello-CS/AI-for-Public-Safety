# visual_intelligence.py
import base64
import cv2
from openai import OpenAI

class VisualIntelligence:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def _to_data_url_b64_jpg(img_bgr) -> str:
        ok, buf = cv2.imencode(".jpg", img_bgr)
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def describe_clothing(self, person_crop_bgr) -> str:
        data_url = self._to_data_url_b64_jpg(person_crop_bgr)
        if not data_url:
            return "Clothing description unavailable (image encoding failed)."

        prompt = (
            "Describe the person's clothing briefly for a security alert email. "
            "Include: top (type + color), bottom (type + color), footwear, and notable accessories "
            "(hat/hood, backpack, mask). If unclear, say 'unclear'. "
            "Return ONE sentence only."
        )

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )

        # responses API convenience:
        return resp.output_text.strip() if hasattr(resp, "output_text") else "Clothing description unavailable."