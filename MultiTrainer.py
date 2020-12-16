#define training loop function
#send model, dataloader, epochs, optimizer, 
import torch
from torch.nn import CrossEntropyLoss, MSELoss

def multi_train_loop(learning_rate,batch_size,labels,model,dataloader,epochs,optimizer,save_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    W=torch.randn((batch_size,labels,1), requires_grad=True,device=device) # update number of weights to match loss
    
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

            criteria=MSELoss(reduction="none") #CrossEntropyLoss()  ########define type of loss function here
            loss=criteria(output, target) #vector since no reduction
            
            #define set of weights to multiply by loss vector then do back propagation
            weighted_loss=torch.sum(loss*W)
            print(weighted_loss)
            weighted_loss.backward(retain_graph=True)
            W-=learning_rate*W.grad
            print(W.item)

            total_loss+=loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 0:
                print(total_loss)
                torch.save(model.state_dict(),save_name+".pt")  
                total_loss=0
