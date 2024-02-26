import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from model import ViT

from config import * 

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "./train/0shape"))  # data_PATH
    
    # print(data_root)

    image_path = os.path.join(data_root)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    tongue_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in tongue_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices_shape.json', 'w') as json_file: ##### JSON path ######
        json_file.write(json_str)

    batch_size = BATCH_SIZE
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    ########VISION TRANSFORMER#######
    net = ViT(image_size=224, patch_size=32, num_classes=NUM_CLASSES, dim=1024, depth=6, heads=16, mlp_dim=2048, channels=3, dropout=0.1, emb_dropout=0.1)
    #################################
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR) # LEARNING RATE

    epochs = EPOCH
    best_acc = 0.0
    if not os.path.exists('./weight/traincase1'):
        os.makedirs('./weight/traincase1')
    save_path = './weight/traincase1/vit_shape.pth'.format(net) # weight path
    train_steps = len(train_loader)

    train_loss = []
    val_acc = []
    start_time = time.time()

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        running_loss /= train_steps
        val_accurate = acc / val_num

        print('[epoch %d] train_loss: %.8f  val_accuracy: %.8f' %
              (epoch + 1, running_loss, val_accurate))
        
        train_loss.append(running_loss)
        val_acc.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print("best acc = %.8f" % best_acc)

    end_time = time.time()  
    execution_time = end_time - start_time  
    print("Execution Time:", execution_time)

    # 绘制训练损失和验证准确率
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.show()

    print('Finished Training') # acc 98%

if __name__ == '__main__':
    main()
