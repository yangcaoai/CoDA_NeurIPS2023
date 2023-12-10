import torch
import clip
from PIL import Image
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, download_root='pretrain_models/', if_transform_tensor=True)

text = clip.tokenize(["bed", "light", "chair", 'end table', 'tissue box', 'book', 'step stool', 'shoe', 'bulletin board', 'fridge', 'frige', 'wall decor']).to(device)
#image = Image.open("CLIP.png")
#image = np.array(image.convert('RGB'))
# image = cv2.imread("../crop_outputs/021_0_pc_input_lvisclasssettings_sunrgbdv2_96seen_1080epoch_gtbox_blank_bg_select/projected_2d_box_region/000135_00001_tissuebox.png")
#print(image.shape)
image = cv2.imread("000006.jpg")
image = np.array(image, np.uint8)
width = image.shape[0]
height = image.shape[1]
input_image_padded = np.ones((730, 730, 3), dtype=np.uint8) * 255

x_offset = (730 - height) // 2
y_offset = (730 - width) // 2

input_image_padded[ y_offset:y_offset + width, x_offset:x_offset + height, :] = image
cv2.imwrite('input_6.png', input_image_padded)

image = torch.from_numpy(image).permute((2, 0, 1)) 
#image = torch.flip(image, dims=[0])
image = preprocess(image).unsqueeze(0).to(device)
#print(torch.max(image))


with torch.no_grad():
    image_features = model.encode_image(image)
    # text_features = model.encode_text(text)
#    print(image_features.shape)
#    print(text_features.shape)
#     logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#    print(probs)

#model, preprocess = clip.load("ViT-L/14", device=device, download_root='pretrain_models/', if_transform_tensor=False)
#image = Image.open("../crop_outputs/021_0_pc_input_lvisclasssettings_sunrgbdv2_96seen_1080epoch_gtbox_blank_bg_select/projected_2d_box_region/000135_00001_tissuebox.png")
#print(image.size)
#image = preprocess(image).unsqueeze(0).to(device)
#print(torch.max(image))



#text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

#with torch.no_grad():
#    image_features = model.encode_image(image)
#    text_features = model.encode_text(text)
#    print(image_features.shape)
#    print(text_features.shape)
#    logits_per_image, logits_per_text = model(image, text)
#    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#    print(probs)



