import os
import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import natsort
import torchvision
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from src.recognition.recognition_helper import RecognitionModel, make_recognition_model
import yaml

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

with open('/id_model/default.yaml', 'r') as file:
    id_dict = yaml.safe_load(file)

with open('/face_model/default.yaml', 'r') as file:
    face_dict = yaml.safe_load(file)

id_config = Config(id_dict)
face_config = Config(face_dict)


face_model: RecognitionModel = make_recognition_model(face_config, enable_training=False)
id_model: RecognitionModel = make_recognition_model(id_config, enable_training=False)

vgg = torchvision.models.vgg16(pretrained=True).features.eval()


def process_value(value):
    if np.isinf(value):
        return 100
    else:
        return value


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = preprocess(image)
    return image_tensor.unsqueeze(0)


def perceptual_loss(vgg, image1, image2):
    features1 = vgg(image1)
    features2 = vgg(image2)
    return F.mse_loss(features1, features2)


def load_images_from_folders(folder_id, folder_style, folder_gen):
    files_id = [f for f in os.listdir(folder_id) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files_style = [f for f in os.listdir(folder_style) if f.endswith(('.png', '.jpg', '.jpeg'))]
    files_gen = [f for f in os.listdir(folder_gen) if f.endswith(('.png', '.jpg', '.jpeg'))]

    files_id = natsort.natsorted(files_id)
    files_style = natsort.natsorted(files_style)
    files_gen = natsort.natsorted(files_gen)

    assert files_id == files_style == files_gen, "Name Match Error"

    image_paths_id = [os.path.join(folder_id, f) for f in files_id]
    image_paths_style = [os.path.join(folder_style, f) for f in files_style]
    image_paths_gen = [os.path.join(folder_gen, f) for f in files_gen]

    return image_paths_id, image_paths_style, image_paths_gen


def batch_evaluate(image_paths_id, image_paths_style, image_paths_gen):
    psnr_id_list, psnr_style_list = [], []
    ssim_id_list, ssim_style_list = [], []
    lpips_id_list, lpips_style_list = [], []
    perceptual_loss_id_list = []
    perceptual_loss_style_list = []
    identity_similarity_list = []
    au_similarity_list = []

    num_images = len(image_paths_id)
    loss_fn = lpips.LPIPS(net='vgg')
    loss_fn.eval()

    for i in range(num_images):

        X_id = Image.open(image_paths_id[i]).convert("RGB")
        X_style = Image.open(image_paths_style[i]).convert("RGB")
        X_gen = Image.open(image_paths_gen[i]).convert("RGB")

        X_id_np = np.array(X_id).astype(np.float64) / 255.0
        X_style_np = np.array(X_style).astype(np.float64) / 255.0
        X_gen_np = np.array(X_gen).astype(np.float64) / 255.0

        psnr_id = process_value(psnr(X_id_np, X_gen_np, data_range=1.0))
        psnr_style = process_value(psnr(X_style_np, X_gen_np, data_range=1.0))
        psnr_id_list.append(psnr_id)
        psnr_style_list.append(psnr_style)

        ssim_id, _ = ssim(X_id_np, X_gen_np, full=True, channel_axis=-1, data_range=1.0)
        ssim_style, _ = ssim(X_style_np, X_gen_np, full=True, channel_axis=-1, data_range=1.0)
        ssim_id_list.append(ssim_id)
        ssim_style_list.append(ssim_style)

        # LPIPS
        X_id_tensor = preprocess_image(X_id)
        X_style_tensor = preprocess_image(X_style)
        X_gen_tensor = preprocess_image(X_gen)

        with torch.no_grad():
            lpips_id = loss_fn(X_id_tensor, X_gen_tensor).item()
            lpips_style = loss_fn(X_style_tensor, X_gen_tensor).item()
            lpips_id_list.append(lpips_id)
            lpips_style_list.append(lpips_style)

        loss_style = perceptual_loss(vgg, X_style_tensor, X_gen_tensor)
        loss_id = perceptual_loss(vgg, X_id_tensor, X_gen_tensor)
        perceptual_loss_style_list.append(loss_style.item())
        perceptual_loss_id_list.append(loss_id.item())
        with torch.no_grad():
            id_id_feature = id_model(X_id_tensor)
            gen_id_feature = id_model(X_gen_tensor)
        identity_similarity = cosine_similarity(id_id_feature.cpu().numpy(), gen_id_feature.cpu().numpy())
        identity_similarity_list.append(identity_similarity[0][0])

        with torch.no_grad():
            style_face_feature = face_model(X_style_tensor)
            gen_face_feature = face_model(X_gen_tensor)
            style_id_feature = id_model(X_style_tensor)
        style_style_feature = style_face_feature - style_id_feature
        gen_style_feature = gen_face_feature - gen_id_feature
        au_similarity = cosine_similarity(style_style_feature.cpu().numpy(), gen_style_feature.cpu().numpy())
        au_similarity_list.append(au_similarity[0][0])


    avg_psnr_id = np.mean(psnr_id_list)
    avg_psnr_style = np.mean(psnr_style_list)
    avg_ssim_id = np.mean(ssim_id_list)
    avg_ssim_style = np.mean(ssim_style_list)
    avg_lpips_id = np.mean(lpips_id_list)
    avg_lpips_style = np.mean(lpips_style_list)
    avg_perceptual_id_loss = np.mean(perceptual_loss_id_list)
    avg_perceptual_style_loss = np.mean(perceptual_loss_style_list)
    avg_identity_similarity = np.mean(identity_similarity_list)
    avg_au_similarity = np.mean(au_similarity_list)

    return {
        'avg_psnr_id': avg_psnr_id,
        'avg_psnr_style': avg_psnr_style,
        'avg_ssim_id': avg_ssim_id,
        'avg_ssim_style': avg_ssim_style,
        'avg_lpips_id': avg_lpips_id,
        'avg_lpips_style': avg_lpips_style,
        'avg_perceptual_id_loss': avg_perceptual_id_loss,
        'avg_perceptual_style_loss': avg_perceptual_style_loss,
        'avg_identity_similarity': avg_identity_similarity,
        'avg_au_similarity': avg_au_similarity,
    }

def main():

    folder_id = "..."
    folder_style = "..."
    folder_gen = "..."

    image_paths_id, image_paths_style, image_paths_gen = load_images_from_folders(folder_id, folder_style, folder_gen)

    results = batch_evaluate(image_paths_id, image_paths_style, image_paths_gen)

    print(f"Average PSNR (ID vs Gen): {results['avg_psnr_id']}")
    print(f"Average PSNR (Style vs Gen): {results['avg_psnr_style']}")
    print(f"Average SSIM (ID vs Gen): {results['avg_ssim_id']}")
    print(f"Average SSIM (Style vs Gen): {results['avg_ssim_style']}")
    print(f"Average LPIPS (ID vs Gen): {results['avg_lpips_id']}")
    print(f"Average LPIPS (Style vs Gen): {results['avg_lpips_style']}")
    print(f"Average Perceptual Loss (ID vs Gen): {results['avg_perceptual_id_loss']}")
    print(f"Average Perceptual Loss (Style vs Gen): {results['avg_perceptual_style_loss']}")
    print(f"Average Identity Similarity: {results['avg_identity_similarity']}")
    print(f"Average AU Similarity: {results['avg_au_similarity']}")


if __name__ == "__main__":
    main()
