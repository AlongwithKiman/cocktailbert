import torch
import numpy as np

def postprocess_model(model, text, tokenizer):
  inputs = tokenizer.batch_encode_plus([text])
  out = model(input_ids = torch.tensor(inputs['input_ids']),
              attention_mask = torch.tensor(inputs['attention_mask']), token_type_ids = torch.tensor(inputs['token_type_ids']))

  size, abv, color = [torch.argmax(logit).item() for logit in out[0]]

  label0 = {
  0: "long",
  1: "short",
  2: "shot"
}

  label1 = {
      0: (0,15),
      1: (15,30),
      2: (30,100)
      }


  label2 = {
    0: "red",
    1: "green",
    2: "blue",
    3: "brown",
    4: "yellow"
  }

  return label0[size], label1[abv], label2[color]



class ColorComparator:

    def __init__(self):
        pass
    
    @staticmethod
    def _rgb_to_lab(rgb):
        num = 0
        RGB = [0, 0, 0]

        for value in rgb:
            value = float(value) / 255

            if value > 0.04045:
                value = ((value + 0.055) / 1.055) ** 2.4
            else:
                value = value / 12.92

            RGB[num] = value * 100
            num = num + 1

        XYZ = [0, 0, 0]

        X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
        Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
        Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
        XYZ[0] = round(X, 4)
        XYZ[1] = round(Y, 4)
        XYZ[2] = round(Z, 4)

        XYZ[0] = float(XYZ[0]) / 95.047
        XYZ[1] = float(XYZ[1]) / 100.0
        XYZ[2] = float(XYZ[2]) / 108.883

        num = 0
        for value in XYZ:
            if value > 0.008856:
                value = value ** (0.3333333333333333)
            else:
                value = (7.787 * value) + (16 / 116)

            XYZ[num] = value
            num = num + 1

        Lab = [0, 0, 0]

        L = (116 * XYZ[1]) - 16
        a = 500 * (XYZ[0] - XYZ[1])
        b = 200 * (XYZ[1] - XYZ[2])

        Lab[0] = round(L, 4)
        Lab[1] = round(a, 4)
        Lab[2] = round(b, 4)

        return Lab
    
    @staticmethod
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))
    
    def color_similarity(self, color1, color2):
        rgb1, rgb2 = self.hex_to_rgb(color1), self.hex_to_rgb(color2)
        lab1, lab2 = self._rgb_to_lab(rgb1), self._rgb_to_lab(rgb2)
        distance = np.linalg.norm(np.array(lab1) - np.array(lab2))
        return distance

# 사용 예시

