from segment_anything import sam_model_registry
from sam_lora_image_encoder import LoRA_Sam
from base_dataset import RadioGalaxyNET
from torchvision.transforms import v2 as T
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
from torchvision.transforms import InterpolationMode
import torch
from tqdm import tqdm


"""
1. Create the SAM-Lora Model
"""
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
rank = 4

sam = sam_model_registry[model_type](
    num_classes = 4,                                                           
    checkpoint=sam_checkpoint,
    image_size = 512)

net = LoRA_Sam(sam[0], 4).cuda()

"""
2. Create the Dataset
"""
train_dir = "RadioGalaxyNET/train"
train_coco = "RadioGalaxyNET/annotations/train.json"
# Note other transformations are not implemented as of this moment.
def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Resize((450, 450), interpolation=InterpolationMode.BILINEAR))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# Define the dataset class
train_dataset = RadioGalaxyNET(root=train_dir,
                          annFile=train_coco,
                          transforms=get_transform(train=True)
                          ) 

"""
3. Create the Dataloader
"""
def collate_fn(batch):
    images = []
    annotations = []
    for img, ann in batch:
        images.append(img)
        annotations.append(ann)
    return images, annotations

# Batch size
train_batch_size = 1

# Define DataLoader for some reason my dataloader can only be = 0, please try different number and let me know how you guys goes. 
trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=0,
                                          collate_fn=collate_fn)

"""
4. Define Loss function
"""
def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['masks']
    #print(low_res_logits.shape)
    #print(low_res_label_batch.shape)
    loss_ce = ce_loss(low_res_logits, low_res_label_batch)
    loss = loss_ce
    #loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    #loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce
#,loss_ce, loss_dice

"""
5. Hyperparameters
"""
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net.to(device)
net.train()
params = [p for p in net.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

"""
5.5 Dealing with the mask problem
"""
def pan_to_sem(label):
    t = torch.empty(1, 5, 450, 450).int()
    masks = label['masks']
    masks = (masks != 0).int()
    classes = label['labels']

    for idx in range(len(classes)):
        # Add the object of a given class into the tensor storing the semantic segmentation mask.
        t[:,classes[idx]] += masks[idx]
        # Convert anything above 0 to 1. 
        t[:,classes[idx]] = (t[:,classes[idx]] > 0).int()
    # Create the background class and set it to 0.
    summed_tensor = torch.sum(t, dim=1)
    t[:,0] = (summed_tensor == 0).int()
    return t

"""
6. Train
"""
iter_num = 0
max_epoch = 2
img_size = 1024
len_dataloader = len(trainloader)
iterator = tqdm(range(max_epoch))
multimask_output = True

ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(4 + 1)
for epoch_num in iterator:
    i = 0   
    for imgs, annotations in trainloader:
        i += 1
        imgs = list(img for img in imgs)
        imgs = torch.stack(imgs, dim=0).to(device)
        #masks_list = [annotation["masks"] for annotation in annotations if "masks" in annotation]
        #masks_list = torch.stack(masks_list, dim=0).to(device)

        # Forward pass
        outputs = net(imgs, multimask_output, img_size)
        mask_gt = pan_to_sem(annotations[0]).float().to(device)

        loss, loss_ce = calc_loss(outputs, mask_gt , ce_loss, dice_loss)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        print(f'Iteration: {i}/{len_dataloader}, Loss: {loss}')