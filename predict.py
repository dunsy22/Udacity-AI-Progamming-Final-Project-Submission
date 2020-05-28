# imports for process_image function
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import time


def arg_parser():
    ''' Receives inputs from user to define image and model checkpoint
        to be loaded
    '''
    parser = argparse.ArgumentParser(description="Parser for prediction algorithm.")
    parser.add_argument('image_path', type=str, help='Path to image file for prediction.', default='flowers/test/1/image_06743.jpg')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file to load prediction model.')
    parser.add_argument('--top_k', type=int, help='Choose how many topk predictions to be displayed.', default=5)
    parser.add_argument('--cat_names', type=str, help='Category names mapping.', default='cat_to_name.json')
    parser.add_argument('--gpu', type=str, help="Defaults to use GPU. Type 'CPU' to switch.", default='gpu')

    args = parser.parse_args()
    
    return args


def load_checkpoint(args):
    checkpoint = torch.load(args.checkpoint)
    
    if checkpoint['arch'] == 'Vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(pil_image)
    
    return img_tensor

def predict(image_path, model, top_k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        
        ps = torch.exp(output)
        top_p, top_class = ps.topk(top_k, dim=1)
        
        idx_to_class = {value: key for key,value in model.class_to_idx.items()}
        probs = [p.item() for p in top_p[0].data]
        classes = [idx_to_class[i.item()] for i in top_class[0].data]
        
    return probs, classes


def imshow(image_path, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def main():
        
    # Get Keyword Args from Parser
    args = arg_parser()
    
    if args.gpu == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = load_checkpoint(args)
    model.to(device)
    model.eval()
    print("Model is ready to go! Here's what the classifier looks like for reference:")
    time.sleep(3)
    print(model.classifier)
    time.sleep(3)
    
    image_path = args.image_path
    top_k = args.top_k
    print("Initializing prediction model...")
    print("Image Path = %s" %image_path)
    print("Loaded Checkpoint = %s" %args.checkpoint)
    print("TopK = %d" %top_k)
    time.sleep(3)
    probs, classes = predict(image_path, model, top_k, device)
    
    cat_names = args.cat_names
    # Load label to name mapping
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    names = [cat_to_name[c] for c in classes]
    print("Prediction Complete. Here's what our model predicted in ranked order!")    
    time.sleep(3)
    
    for i in range (top_k):
         print("Number: {}/{}.. ".format(i+1, top_k),
                "Class name: {}.. ".format(names [i]),
                "Probability: {:.3f}..% ".format(probs [i]*100),
                )
    
if __name__ == '__main__': main()