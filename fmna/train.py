import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def train(dataset, model, model_name, num_epochs):
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                           batch_size=512, 
                                           shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)   

    losses = []

    for epoch in range(num_epochs):
        epoch_loss= []
        model.train()
        for d in train_loader:
            d = d.to(device)

            output = model(d)
            loss = criterion(output, d)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, sum(epoch_loss)/len(epoch_loss)))
        losses.append(sum(epoch_loss)/len(epoch_loss))
        if ((epoch+1) % 25 == 0) and epoch != 0:
            print('save')
            torch.save(model.state_dict(), model_name + '_' + str(epoch + 1) + '.pth')
    
    # Save the model
    torch.save(model.state_dict(), model_name + '_final' + '.pth')

    # Save loss grafic
    fig, ax = plt.subplots()
    ax.plot(losses)
    plt.title("Loss")
    plt.savefig(model_name + '.png')

    return losses



    



