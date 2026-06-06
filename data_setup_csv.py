import os
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from torchvision.io import decode_image

def find_classes(img_label):
      """  should contain a column named 'label' which indicate the class of the dataset """
      classes = sorted(img_label.iloc[:, -1].unique())
      class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
      return classes,class_to_idx

class ImageFolderCSV(Dataset):
      def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform
            self.classes , self.class_to_idx = find_classes(self.img_labels)
      def __len__(self):
            return len(self.img_labels)

      def __getitem__(self,idx):
            img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
            image = decode_image(img_path)
            label = self.img_labels.iloc[idx,-1]
            label = self.class_to_idx[label]
            if self.transform:
                  image = self.transform(image)
            if self.target_transform:
                  label = self.target_transform(label)

            return image,label

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
            train_dir: str, 
            test_dir: str,
            train_annotations_file:str, 
            test_annotations_file:str,
            transform: transforms.Compose, 
            batch_size: int, 
            num_workers: int=NUM_WORKERS):
      
      train_data = ImageFolderCSV(train_annotations_file,train_dir,transform)
      test_data = ImageFolderCSV(test_annotations_file,test_dir,transform)

      class_names = train_data.classes

      train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
      )

      test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
      )

      return train_dataloader, test_dataloader, class_names

