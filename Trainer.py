#define training loop function
#send model, dataloader, epochs, optimizer, 


def train_loop(model,dataloader,epochs,optimizer):
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        train_correct = 0
        data=iter(dataset_loader)
        for sample, target in data:
            print(target)








"""
def train(optimizer, epoch):
    
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    for batch_index, batch_samples in enumerate(train_loader):
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

        optimizer.zero_grad()
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        train_loss += criteria(output, target.long())


        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
    
        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
"""