import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import torch

feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

image = Image.open('foot.jpg')
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height%32) 
new_width = int(new_height*image.width/image.height)
diff = new_width % 32
new_width = new_width-diff if diff<16 else new_width+32-diff
new_size=(new_width,new_height)
image=image.resize(new_size)

inputs=feature_extractor(images=image,return_tensors="pt")

with torch.no_grad():
    outputs=model(**inputs)
    predicted_depth=outputs.predicted_depth
    
pad = 16
output = predicted_depth.squeeze().cpu().numpy()*1000.0
output = output[pad:-pad,pad:-pad]
image=image.crop((pad, pad, image.width-pad, image.height-pad))

fig, ax=plt.subplots(1,2)
ax[0].imshow(image)
ax[0].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax[1].imshow(output,cmap='plasma')
ax[1].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
plt.tight_layout()
plt.pause(5)

import numpy as np
import open3d as o3d

width, height=image.size
depth_image=(output*255/np.max(output)).astype('uint8')
image=np.array(image)

depth_o3d=o3d.geometry.Image(depth_image)
image_o3d=o3d.geometry.Image(image)
rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d,depth_o3d,convert_rgb_to_intensity=False)

camera_intrinsic=o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width,height,500,500,width/2,height/2)

pcd=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_intrinsic)
o3d.visualization.draw_geometries([pcd])