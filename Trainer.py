#define training loop function
#send model, dataloader, epochs, optimizer, 
import torch
from torch.nn import CrossEntropyLoss

def train_loop(model,dataloader,epochs,optimizer,save_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(1, epochs+1):
        model.train()
        total_loss=0

        data=iter(dataloader)
        batch_index=0
        for batch in data:
            batch_index+=1
            #print(batch_index)
            #print(batch)
            sample, target = batch

            sample=sample.to(device)
            target=target.to(device)
            #print(target)
            #print(sample.size())
            output=model(sample)

            criteria=CrossEntropyLoss()  ########define type of loss function here
            loss=criteria(output, target)
            
            total_loss+=loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 0:
                print(total_loss)
                torch.save(model.state_dict(),save_name+".pt")  
                total_loss=0
