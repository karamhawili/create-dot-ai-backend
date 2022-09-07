'''
dnnlib/   torch_utils/   training/   legacy.py

The above files/directories are taken from the official stylegan2-ada-pytorch repository:
                                      https://github.com/NVlabs/stylegan2-ada-pytorch.git

'''


try:
    import unzip_requirements
except ImportError:
    pass


import base64
from io import BytesIO
import json
import boto3
import uuid
import dnnlib
import numpy as np
import PIL.Image
import torch
import random
import legacy

#_____________________________________________________ helper functions

# the "latent vector" in this file is referred to as "z"

def imgs_to_base64_str(imgs):
    print('Converting images into base64 strings...')
    str_imgs = []
    for img in imgs:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_byte = buffered.getvalue()
        img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
        str_imgs.append(img_str)
    return str_imgs


def load_model():
    print('Loading networks from "%s"...' % download_path)
    device = torch.device('cpu')
    with dnnlib.util.open_url(download_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    return G


def generate_images_from_seeds(
    G,
    truncation_psi: float,
    noise_mode: str,
    seeds = None,
    number_of_images = None,
):
  label = torch.zeros([1, G.c_dim], device='cpu')

  if seeds is None and number_of_images is None:
    raise ValueError('Expected either a list of seeds or a number of images. Neither is provided.')

  if seeds is not None and number_of_images is not None:
    raise ValueError('Expected either a list of seeds or a number of images, not both.')

  imgs = []

  if seeds is not None:
    print('seeds param is not none', seeds)
    # generate images from list of seeds
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds) - 1))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to('cpu')
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        imgs.append(img)
  
  if number_of_images is not None:
    print('Numbers of images to generate:', number_of_images)
    # generate seeds
    seeds = random.sample(range(0, 2**32 - 1), number_of_images)
    print('Seeds', seeds)
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds) - 1))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to('cpu')
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        imgs.append(img)


  return imgs, seeds


def get_z_from_seed(
    G,
    seed: int,
):
  return torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to('cpu')


def linear_interpolate(z1, z2, alpha):
  return z1 * alpha + z2 * (1 - alpha)


def generate_image_from_z(
    G,
    z: torch.Tensor,
    truncation_psi: float,
    noise_mode: str,
):
  label = torch.zeros([1, G.c_dim], device='cpu')
  img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
  return img


def steer_output(
    G,
    z_output,
    seed_target,
    truncation_psi: float,
    noise_mode: str,
    alpha: float,
):
  # first, convert seed_target to z_target
  z_target = get_z_from_seed(G, seed_target)
  
  # interpolate with respect to alpha
  z_out_new = linear_interpolate(z_target, z_output, alpha) 
  # the higher the alpha the more the output will diverge from the current design to the target design


  label = torch.zeros([1, G.c_dim], device='cpu')
  # generate image from vector
  img = G(z_out_new, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
  return img, z_out_new



def get_image_from_targets(
    G,
    zs,
    truncation_psi: float,
    noise_mode: str,
):
  if len(zs) < 2:
    print('Provide at least 2 design targets')
    return
  
  # find latent vector equidistant from all targets
  z_eq = sum(zs) / 4

  label = torch.zeros([1, G.c_dim], device='cpu')
  img = G(z_eq, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
  return img, z_eq


#___________________________________________________________________________ end of helper functions


# load network from s3 bucket
s3_client = boto3.client('s3')

bucket = 'bucketformlmodels'
key = 'network-snapshot-001280.pkl' 
download_path = '/tmp/{}'.format(key)

print('Downloading file {} from {}'.format(key, bucket))
s3_client.download_file(bucket, key, download_path)


def lambda_handler(event, context):

    if event.get("source") == "aws.events":
        print("Lambda is warm!")
        return {}

    '''
    - actions:
        - generate_random
        - generate_output
        - interpolate
        - get_image_from_z
        - generate_from_seeds
    
    - generate_random
        - request   :   action, truncation_psi, noise_mode, number_of_images
        - response  :   list[images], List[seeds]

    - generate_output
        - request   :   action, List[seeds], truncation_psi, noise_mode
        - response  :   list[images], List[z_out]

    - interpolate
        - request   :   action, List[z_out], seed_target, truncation_psi, noise_mode, alpha
        - response  :   list[images], List[z_out]

    - get_image_from_z
        - request   :   action, List[z_out], truncation_psi, noise_mode
        - response  :   list[images]

    - generate_from_seeds
        - request   :  action List[seeds], truncation_psi, noise_mode
        - response  :  list[images], List[seeds]
    '''

    print('Inside lambda handler...')

    # load model
    G = load_model()

    # initialize variables
    imgs = []
    seeds_list = []
    z_out = []
    
    print('event json object received as ', event)
    
    data = json.loads(event["body"])

    print('User requested to {}'.format(data["action"]))
    print('Data received from user: {} {}'.format(float(data["truncation_psi"]), data["noise_mode"]))

    # generate images
    if data["action"] == "generate_from_seeds":
        print('Generating images from seeds ', data["seeds"])
        imgs, seeds_list = generate_images_from_seeds(G, truncation_psi=float(data["truncation_psi"]), noise_mode=data["noise_mode"], seeds=data["seeds"], number_of_images=None)

    elif data["action"] == "generate_random":
        print('Generating {} random images'.format(int(data["number_of_images"])))
        imgs, seeds_list = generate_images_from_seeds(G, truncation_psi=float(data["truncation_psi"]), noise_mode=data["noise_mode"], number_of_images=int(data["number_of_images"]), seeds=None)

    elif data["action"] == "generate_output":
        # get design targets seeds
        seeds_list = data["seeds"]
        print("Generating output from {} design targets".format(len(seeds_list)))
        zs = []
        for seed in seeds_list:
            zs.append(get_z_from_seed(G, seed))
        img, z_out = get_image_from_targets(G, zs, float(data["truncation_psi"]), data["noise_mode"])
        imgs.append(img)
        # convert latent vector to list
        z_out = z_out.tolist()

    elif data["action"] == "interpolate":
        print('Interpolating between output and design target...')
        # convert vector from list to tensor
        z_out_tensor = torch.Tensor(json.loads(data["z_out"])).cpu()
        img, z_out = steer_output(G, z_out_tensor, int(data["seed_target"]), float(data["truncation_psi"]), data["noise_mode"], float(data["alpha"]))
        imgs.append(img)
        z_out = z_out.tolist()

    elif data["action"] == "get_image_from_z":
        print('Getting image from latent vector...')
        # convert vector from list to tensor
        z_out_tensor = torch.Tensor(json.loads(data["z_out"])).cpu()
        img = generate_image_from_z(G, z_out_tensor, float(data["truncation_psi"]), data["noise_mode"])
        imgs.append(img)

        
    print("Generated {} images from {} seeds".format(len(imgs), len(seeds_list)))


    # from base64 to str
    str_imgs = imgs_to_base64_str(imgs)


    '''
    The images key inside body either
            contains all images generated for design targets
            or, output equidistant from all targets
            or, output from a latent space interpolation
    '''
    body = {
        "images": str_imgs,
        "seeds": seeds_list,
        "z_out": z_out
    }


    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        }

    }

    return response

