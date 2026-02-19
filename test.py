import os
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
#from datasetpcagpc import SEN12MS, DFC2020
#from datasetpcagpcnos2 import SEN12MS, DFC2020
from datasetfusiont import SEN12MS, DFC2020
from models.deeplab import DeepLab
from models.unet import UNet
from models.at_unet import UNet_Attention
from models.cbamunet import CBAM_UNet
from models.segformer import SegFormer
from models.vit import ViT
#from models.transunet import ViT_UNet
from models.vitencoder import ViT_UNet
from utils import labels_to_dfc
from utils import mycmap as dfc_cmap
from utils import mypatches as dfc_legend

# define and parse arguments
parser = argparse.ArgumentParser()

# config
parser.add_argument('--config_file', type=str, default="/workspace/sen12ms/dfc/code/12_08_2024logs/DATA-ablation-CBAM/experiment_CBAM_only_S1/checkpoints/args.pkl",
                    help='path to config file (default: ./args.conf)')
parser.add_argument('--checkpoint_dir', type=str, default="/workspace/sen12ms/dfc/code/12_08_2024logs/DATA-ablation-CBAM/experiment_CBAM_only_S1/checkpoints/",
                    help='path to directory containing checkpoint files')

# general
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for prediction (default: 32)')
parser.add_argument('--workers', type=int, default=4,
                    help='number of workers for dataloading (default: 4)')
parser.add_argument('--score', action='store_true', default=True,
                    help='score prediction results using ground-truth data')

# data
parser.add_argument('--dataset', type=str, default="dfc2020_test",
                    choices=['sen12ms_holdout', 'dfc2020_val', 'dfc2020_test'],
                    help='type of dataset (default: sen12ms_holdout)')
parser.add_argument('--data_dir', type=str, default="/workspace/sen12ms",
                    help='path to dataset')
parser.add_argument('--preview_dir', type=str, default="/workspace/sen12ms/dfc/code/12_08_2024prediction/CBAM-DATA-ABLATION/onlyS1",
                    help='path to preview dir (default: no previews)')

args = parser.parse_args()

# Load config
train_args = pkl.load(open(args.config_file, "rb"))

# Create preview dir
if args.preview_dir is not None:
    os.makedirs(args.preview_dir, exist_ok=True)

# Set flags for GPU processing if available
if torch.cuda.is_available():
    args.use_gpu = True
    if torch.cuda.device_count() > 1:
        raise NotImplementedError("multi-gpu prediction not implemented! "
                                  + "try to run script as: "
                                  + "CUDA_VISIBLE_DEVICES=0 predict.py")
else:
    args.use_gpu = False

# load dataset
if args.dataset == "sen12ms_holdout":
    dataset = SEN12MS(args.data_dir,
                      subset="holdout",
                      no_savanna=train_args.no_savanna,
                      use_s2hr=train_args.use_s2hr,
                      use_s2mr=train_args.use_s2mr,
                      use_s2lr=train_args.use_s2lr,
                      #use_pca=train_args.use_pca,
                      #use_fusiont=train_args.use_fusiont,
                      #use_gpc=train_args.use_gpc,
                      #use_hr=train_args.use_hr,
                      use_s1=train_args.use_s1)
    gt_id = "lc"
#elif args.dataset == "tiff_dir":
#    #assert not args.score
#    dataset = TIFFDir(args.data_dir,
#                      no_savanna=train_args.no_savanna,
#                      use_s2hr=train_args.use_s2hr,
#                      use_s2mr=train_args.use_s2mr,
#                      use_s2lr=train_args.use_s2lr,
#                      use_pca=train_args.use_pca,
#                      use_s1=train_args.use_s1)
#    gt_id = "pred"
else:
    dfc2020_subset = args.dataset.split("_")[-1]
    dataset = DFC2020(args.data_dir,
                      subset=dfc2020_subset,
                      no_savanna=train_args.no_savanna,
                      use_s2hr=train_args.use_s2hr,
                      use_s2mr=train_args.use_s2mr,
                      use_s2lr=train_args.use_s2lr,
                      #use_pca=train_args.use_pca,
                      #use_fusiont=train_args.use_fusiont,
                      #use_gpc=train_args.use_gpc,
                      #use_hr=train_args.use_hr,
                      use_s1=train_args.use_s1)
    gt_id = "dfc"
n_classes = dataset.n_classes
n_inputs = dataset.n_inputs

# Set up dataloader
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=False)

# Set up the model
if train_args.model == "deeplab":
    model = DeepLab(num_classes=n_classes, backbone='resnet', pretrained_backbone=False,
                    output_stride=train_args.out_stride, sync_bn=False, freeze_bn=False,
                    n_in=n_inputs)
#else:
#    model = UNet(n_classes=n_classes, n_channels=n_inputs)
else:
    model = CBAM_UNet(n_classes=n_classes, n_channels=n_inputs)
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    #model = MixFormer(num_classes=11, in_channels=3)
    #model = SegFormer(in_channels=n_inputs, num_classes=n_classes)
    #model = ViT(in_channels=n_inputs, num_classes=n_classes)
    #model = ViT_UNet(num_classes=n_classes, in_channels=n_inputs)
    initialize_weights(model) 
    
    
if args.use_gpu:
    model = model.cuda()

# Iterate over each checkpoint file in the directory
checkpoint_dir = args.checkpoint_dir
checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.endswith(".pth")]

for checkpoint_file in checkpoint_files:
    # Load the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    state = torch.load(checkpoint_path)
    step = state["step"]
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("Loaded checkpoint from step", step)

    # Initialize scoring if ground-truth is available
    if args.score:
        import metrics
        conf_mat = metrics.ConfMatrix(n_classes)

    # Predict samples
    for batch in tqdm(dataloader, desc="[Pred]"):
        # Unpack sample
        image = batch['image']
        if args.score:
            target = batch['label']

        # Move data to GPU if model is on GPU
        if args.use_gpu:
            image = image.cuda()
            if args.score:
                target = target.cuda()

        # Forward pass
        with torch.no_grad():
            prediction = model(image)

        # Convert to 256x256 numpy arrays
        prediction = prediction.cpu().numpy()
        prediction = np.argmax(prediction, axis=1)
        if args.score:
            target = target.cpu().numpy()

        # Save predictions
        for i in range(prediction.shape[0]):
            id = batch["id"][i].replace("_s2_", "_" + gt_id + "_")
            output = labels_to_dfc(prediction[i, :, :], train_args.no_savanna)
            output = output.astype(np.uint8)
            output_img = Image.fromarray(output)
            # output_img.save(os.path.join(args.out_dir, id))

            # Update error metrics if scoring is enabled
            if args.score:
                gt = labels_to_dfc(target[i, :, :], train_args.no_savanna)
                conf_mat.add(target[i, :, :], prediction[i, :, :])

    # Print scoring results
    if args.score:
        print("AA\t", conf_mat.get_aa())
        print("mIoU\t", conf_mat.get_mIoU())
        print("IoU\t", conf_mat.get_IoU())
        print(conf_mat.calc(target, prediction))

    # Calculate confusion matrix
    cm = conf_mat.calc(target, prediction)
    
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt
    
    def plot_confusion_matrix1(cm, classes,
                               normalize=True,
                               title='Confusion Matrix',
                               cmap=plt.cm.Greys):
    
        plt.imshow(np.ones_like(cm), interpolation='nearest', cmap=plt.get_cmap('binary'))
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=60)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)  # Avoid division by zero
            print("Confusion Matrix (Normalized)")
        else:
            cm_norm = cm.astype('float')
            print('Confusion matrix (Unnormalized)')
    
        thresh = cm_norm.max() / 2.
        for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
            plt.text(j, i, "{:.4f}".format(cm_norm[i, j]), horizontalalignment="center", color="black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Assuming cm is your confusion matrix
    cm = conf_mat.calc(target, prediction)
    
    # Calculate overall accuracy
    accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
    
    # Calculate producer accuracy
    producer_accuracy = cm[0, 0] / np.sum(cm[0, :])
    
    # Calculate user accuracy
    user_accuracy = cm[0, 0] / np.sum(cm[:, 0])
    
    # Calculate expected accuracy
    expected_accuracy = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / np.sum(cm)**2
    
    # Calculate kappa coefficient
    kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy)
    
  
#    # Calculate precision, recall, and F1 score
#    precision = precision_score(target, prediction)
#    recall = recall_score(target, prediction)
#    f1 = f1_score(target, prediction)
    
    # Calculate accuracy for each class
    class_accuracies = np.diagonal(cm) / np.sum(cm, axis=1)
    
    # Calculate average accuracy
    average_accuracy = np.mean(class_accuracies)

    
    print("Overall accuracy:", accuracy)
    print("Producer accuracy:", producer_accuracy)
    print("User accuracy:", user_accuracy)
    print("Expected accuracy:", expected_accuracy)
    print("Kappa coefficient:", kappa)
    #print("Precision:", precision)
    #print("Recall:", recall)
    #print("F1 score:", f1)
    print("Average accuracy:", average_accuracy)

        
    
    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_confusion_matrix1(cm, classes=['Forest', 'Shrubland', 'Grassland', 'Wetland', 'Cropland',
                                        'Urban/Builtup', 'Snow/Ice', 'Barren', 'Water'])
    fig.savefig(os.path.join(args.preview_dir, "cm_" + str(conf_mat.get_aa()).replace("0.", "_TEST(2986)_") + "_" + os.path.splitext(checkpoint_file)[0] + ".png"))