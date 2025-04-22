
import torch
import time

class Average(object):
    """
    calculates average and current values
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.val = 0
        self.sum = 0
        self.avg = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    """
    calculates accuracy
    """
    #get batch
    batch_size = target.shape[0]
    #get prediction
    max, prediction = torch.max(output, dim=-1)
    #num correct
    correct = prediction.eq(target).sum() * 1.0
    #accuracy
    accuracy = correct / batch_size

    return accuracy


def train(epoch, data_loader, model, optimizer, criterion):

    losses = Average()
    acc = Average()

    for idx, (data, target) in enumerate(data_loader):

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        #forward
        out = model(data)
        # print(f'output: ',out.shape)

        #loss
        loss = criterion(out, target)
        # print(f'loss: ',loss.shape)

        #backward
        loss.backward()

        #optimizer
        optimizer.step()
        
        #modified and reused form DL class
        batch_acc = accuracy(out, target)
        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        #calculate stats  modified and reused form DL class
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                   .format(epoch, idx, len(data_loader), loss=losses, top1=acc))
            
def validate(epoch, val_loader, model, criterion):
    losses = Average()
    acc = Average()

    num_class = 11
    confusion_matrix =torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()


        #forward
        out = model(data)
        
        #loss
        with torch.no_grad():
            loss = criterion(out, target)
        

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        #modified and reused from DL class project
        # calculate stats
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
               'Prec @1 {top.val:.4f} ({top.avg:.4f})\t')
               .format(epoch, idx, len(val_loader), loss=losses, top=acc))
            
    #calculate stats  modified and reused from DL class
    confusion_matrix = confusion_matrix / confusion_matrix.sum(1)
    per_cls_acc = confusion_matrix.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("Accuracy: {top.avg:.4f}".format(top=acc))
    return acc.avg, confusion_matrix

def adjust_learning_rate(optimizer, epoch, args):
    #modified and influenced from DL class project
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


