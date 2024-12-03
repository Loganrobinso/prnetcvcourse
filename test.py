from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score
import torch
import argparse
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
from torchmetrics.classification import JaccardIndex

def save_mask_plot(gt_mask, pred_mask, save_path):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth')
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Prediction')
    plt.imshow(pred_mask, cmap='gray')
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()


validation_miou = JaccardIndex(task='multiclass', threshold=0.5, num_classes=30, average='macro').cuda()
per_class_validation_miou = JaccardIndex(task='multiclass', threshold=0.5, num_classes=30, average=None).cuda()

def validate(model, test_loader, proto_features, device, index):
    model.eval()
    
    label_list, gt_mask_list, score_list = [], [], []
    progress_bar = tqdm(total=len(test_loader))
    progress_bar.set_description(f"Evaluating")
    jaccard_list = []
    
    save_dir = '/app/Documents/PRNet/validation_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    samples_to_plot = 5
    
    for step, batch in enumerate(test_loader):
        progress_bar.update(1)
        image, label, mask, _, _ = batch
            
        gt_mask_list.append(mask.squeeze(1).cpu().numpy().astype(bool))
        label_list.append(label.cpu().numpy().astype(int).ravel())
        
        image = image.to(device)
        mask = mask.to(device) 
        
        with torch.no_grad():
            logits = model(image, proto_features)

        validation_miou.update(logits, mask.squeeze(1))
        per_class_validation_miou.update(logits, mask.squeeze(1))
        
        if step == 0:
            step_ = index + 1
            for i in range(samples_to_plot):
                filename = f"pred_{step_}.png"
                step_ += 1
                filepath = os.path.join(save_dir, filename)
                gt_mask = mask[i].squeeze(0).cpu().numpy().astype(int)
                pred_mask = logits[i].argmax(dim=0).cpu().numpy()
                save_mask_plot(gt_mask, pred_mask, filepath)

    progress_bar.close()
    
    val_miou = validation_miou.compute().cpu().numpy()
    val_pc_miou = per_class_validation_miou.compute().cpu().numpy()
    
    pcm_values = [round(miou_value, 4) for miou_value in val_pc_miou]

    validation_miou.reset()
    per_class_validation_miou.reset()

    return val_miou, pcm_values

def load_prototype_features(prototype_path, class_name, device):
    file_path = f"{prototype_path}/{class_name}my_data.npy"
    
    # Check if the file exists and is not empty
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prototype features file not found: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Prototype features file is empty: {file_path}")
    
    try:
        features = torch.load(file_path, map_location=device)
        if not features:
            raise ValueError("Loaded prototype features are empty.")
        return features.to(device)
    except (EOFError, ValueError, FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading prototype features: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a PyTorch checkpoint on a single image."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the PyTorch checkpoint file")
    parser.add_argument("--image", required=True, help="Path to the input image file")
    parser.add_argument("--gt_mask", help="Path to the ground truth mask file")
    parser.add_argument("--prototype_path", required=True, help="Path to the class-specific prototype folder")
    parser.add_argument("--class_name", required=True, help="Class name for prototype features")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    img_size = 256
    crp_size = 256

    transform = T.Compose(
        [
            T.Resize(img_size, Image.LANCZOS),
            T.CenterCrop(crp_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Load and preprocess the image
    image = Image.open(args.image).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Load prototype features
    proto_features = load_prototype_features(args.prototype_path, args.class_name, device)

    # Initialize the model and load checkpoint
    model = PRNet("resnet18", num_classes=1, device=device).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(image, proto_features)
    scores = torch.sigmoid(logits)
    pred_mask = (scores > 0.5).float().cpu().squeeze().numpy()  # Move to CPU and convert to NumPy array

    # Plot the predicted mask
    plt.figure()
    plt.imshow(pred_mask, cmap='viridis')
    plt.title('Predicted Mask')
    plt.colorbar()
    plt.show()

    if args.gt_mask:
        # Load and preprocess the ground truth mask
        gt_mask = Image.open(args.gt_mask).convert("L")
        gt_mask = gt_mask.resize((crp_size, crp_size), Image.NEAREST)
        gt_mask = np.array(gt_mask) / 255.0  # Normalize to [0, 1]

        # Convert to tensor
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float().to(device)  # Add batch and channel dimensions

        # Optionally compare gt_mask with pred_mask or visualize it
        plt.figure()
        plt.imshow(gt_mask.squeeze().cpu().numpy(), cmap='gray')
        plt.title('Ground Truth Mask')
        plt.colorbar()
        plt.show()

