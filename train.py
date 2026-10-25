import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset
from model import CiFAR10

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
BATCH_SIZE = 400
EPOCHS = 10
MAX_LR = 0.01

norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
train_tf = T.Compose([T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip(), T.ToTensor(), norm])
test_tf = T.Compose([T.ToTensor(), norm])

full_train = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tf)
full_val = datasets.CIFAR10(root="data", train=True, download=True, transform=test_tf)
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tf)

perm = torch.randperm(len(full_train)).tolist()
train_data = Subset(full_train, perm[:45000])
val_data = Subset(full_val, perm[45000:])

train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(val_data, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = CiFAR10().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=MAX_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    epochs=EPOCHS,
    steps_per_epoch=len(train_dataloader)
)

def train_step(model: nn.Module,
               data_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler,
               device: torch.device = device):

    train_loss, correct = 0, 0
    model.train()
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        train_loss += loss.item()
        correct += (y_pred.argmax(1) == y).sum().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()

    train_loss /= len(data_loader)
    train_acc = correct / len(data_loader.dataset) * 100

    return train_loss, train_acc

def test_step(model: nn.Module,
              data_loader: DataLoader,
              loss_fn: nn.Module,
              device: torch.device = device):

    test_loss, correct = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)
            test_loss += loss_fn(test_pred, y).item()
            correct += (test_pred.argmax(1) == y).sum().item()

    test_loss /= len(data_loader)
    test_acc = correct / len(data_loader.dataset) * 100

    return test_loss, test_acc

best_val_acc = 0.0
for epoch in range(EPOCHS):

    print(f"\nEpoch: {epoch + 1} / {EPOCHS}")

    train_loss, train_acc = train_step(
        model=model,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    val_loss, val_acc = test_step(
        model=model,
        data_loader=val_dataloader,
        loss_fn=loss_fn,
        device=device
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best.pt")

    print(
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_acc:.2f}% | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_acc:.2f}%"
    )

model.load_state_dict(torch.load("best.pt", map_location=device))
test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
print(f"Test accuracy: {test_acc:.2f}%")