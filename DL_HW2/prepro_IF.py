import torch
from torchvision import transforms, datasets



data_transform = transforms.Compose([
        transforms.Resize(280),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
    ])
hymenoptera_dataset = datasets.ImageFolder(root='animal-10/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

for i_batch, sample_batched in enumerate(dataset_loader):
    print(i_batch, sample_batched[0].size())