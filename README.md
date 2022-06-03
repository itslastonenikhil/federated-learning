# Report: Federated Learning

### Nikhil Maurya - 1901104

### Indian Institute of Information Technology, Guwahati

> The biggest obstacle to using advanced data analysis isn’t skill base or technology; it’s plain old access to the data ~ *Edd Wilder-James, Harvard Business Review*
> 

![img2.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/img2.png)

Modern devices have access to a wealth of data suitable for learning models, which can improve the user experience on the device. However, this rich data is often privacy sensitive, large in quantity, or both.

This problem can be solved using Federated Learning.

**Federated Learning** includes,

1. Training data distributed on the devices.
2. Learning a shared model by aggregating locally computed updates.

**Main Features of Federated Learning**

Federated learning comprises multiple client-server interactions. In each round, the server transmits the current global model to a set of nodes. These nodes train the transmitted model locally and send it to the server. Thereafter, the server aggregates these local models and updates the global model.

A single server round can be summarized with the following steps:

1. Initialization of weights
2. Selection of clients
3. Sending global-model to selected clients
4. Clients training the model on local data.
5. Reporting the locally trained model to the server.
6. Aggregation of local updates
7. Updating global model.

## Federated Learning Algorithms

### FedAvg

**FedAvg** algorithm, combines local stochastic gradient descent (SGD) on each client with and a server that performs model averaging

![[https://arxiv.org/pdf/1602.05629.pdf](https://arxiv.org/pdf/1602.05629.pdf)](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/fedavg.png)

[https://arxiv.org/pdf/1602.05629.pdf](https://arxiv.org/pdf/1602.05629.pdf)

![[https://fedbiomed.gitlabpages.inria.fr/](https://fedbiomed.gitlabpages.inria.fr/)](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/fl-graph.png)

[https://fedbiomed.gitlabpages.inria.fr/](https://fedbiomed.gitlabpages.inria.fr/)

The server executes the **FedAvg** algorithm for several rounds of training.

1. First, the weights are initialized
2. It samples a fraction of ***C*** of the ***K*** clients.
3. The server sends the current round of weights to each client ***k.***
4. The weights for round ***t*** is denoted by ***wt***.
5. The client runs stochastic gradient descent (SDG) on their local data for ***E epochs*.**
6. The updated weights are sent back to the server.
7. Once the server receives the updated weights, it takes the weighted average.

**Problems with FedAvg Algorithm**

1. It assumes all the devices will complete all **E epochs.** But in practice, different devices have different hardware capabilities. Therefore devices with incomplete epochs can lower the rate of convergence.
2. FedAvg takes the weighted average depending upon the amount of data samples in a particular device. Hence it may favor certain devices more than the others.

### FedProx

FedProx is generalization and re-parametrization of FedAvg

In the context of systems heterogeneity, FedAvg does not allow participating devices to perform variable amounts of local work based on their underlying systems constraints.

Instead it is common to simply drop devices that fail to compute **E epochs** within a specified time window.

FedProx solves the above problems by following ideas:

1. Allowing for variable amounts of work to be performed on local devices to handle stragglers
2. Modified local subproblem

FedProx adds a proximal term to the local subproblem to effectively limit the impact of variable local updates.

![fedprox-term.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/fedprox-term.png)

The value of `mu` is selected by hyper-parameter tuning.

![[https://arxiv.org/pdf/1812.06127.pdf](https://arxiv.org/pdf/1812.06127.pdf)](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/fedprox.png)

[https://arxiv.org/pdf/1812.06127.pdf](https://arxiv.org/pdf/1812.06127.pdf)

The proximal term is beneficial in two aspects:

1. It addresses the issue of statistical heterogeneity by restricting the local updates to be closer to the initial (global) model without any need to manually set the number of local epochs.
2. It allows for safely incorporating variable amounts of local work resulting from systems heterogeneity.

### qFedAvg

qFedAvg tries to solve the problem of fair resource allocation of learning resources.

Fairness of performance distribution: For trained models $w$ and $w'$, we informally say that model $w$ provides a more fair solution to the federated learning objective than model $w'$ if the performance of model $w$ on the m devices, {$a_1,...a_m$}, is more uniform than the performance of model $w'$ on the m devices.

A natural idea to achieve fairness would be to reweight the objective assigning higher weights to devices with poor performance, so that the distribution of accuracies in the network shifts towards more uniformity

For given local non-negative cost functions $F_k$ and parameter q > 0, we define the q-Fair Federated Learning (q-FFL) objective as:

![qfedavg-term.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/qfedavg-term.png)

**qFedAvg Algorithm**

![[https://arxiv.org/pdf/1905.10497.pdf](https://arxiv.org/pdf/1905.10497.pdf)](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/qfedavg.png)

[https://arxiv.org/pdf/1905.10497.pdf](https://arxiv.org/pdf/1905.10497.pdf)

# Implementation of Federated Algorithms

- Language: Python
- Machine Learning Framework: PyTorch
- Environment: Google Colab

![python.jpeg](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/python.jpeg)

![google-colab.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/google-colab.png)

![pytorch.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/pytorch.png)

### About Dataset

![MnistExamples.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/MnistExamples.png)

**MNIST** Handwritten Digit Classification Dataset The MNIST dataset is an acronym that stands for the *Modified National Institute of Standards and Technology dataset*.

It is a dataset of 60,000 small square `28x28` pixel grayscale images of handwritten single digits between 0 and 9. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

**Synthetic Digits** with noisy backgrounds for testing robustness of classification algorithms. This dataset contains synthetically generated images of English digits. The images are generated with varying scales and rotations.

**Dataset Loading**

Dataset is loaded from the `torchvision.datasets` module provided by Torchvision. I have loaded the train and test datasets and split them into multiple clients.

- Train sample size per client: 500 samples
- Test sample size per client: 100 samples
- Batch size: 25 samples
- Number of clients: 10

**Client Data Distribution**

Client data is distributed into two types:

1. Independent and identically distributed **(IID)**
2. Non independent and identically distributed **(Non IID)**

**IID MNIST Clients**

![iid1.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iid1.png)

![iid2.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iid2.png)

![iid3.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iid3.png)

**Non IID MNIST Client**

![niid.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid.png)

![niid2.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid2.png)

![niid3.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid3.png)

**Synthetic IID Client**

![iids1.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iids1.png)

![iids2.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iids2.png)

![iids3.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iids3.png)

**Synthetic Non IID Clients**

![niids1.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niids1.png)

![niids2.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niids2.png)

![niids3.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niids3.png)

### Classification

I have defined a Logistic Regression Classifier for digits classification.

```cpp
num_features = 28*28
num_classes = 10

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        output = self.linear(x)
        return output

model = LogisticRegression()
```

### Functions for Federated Algorithms

A class to store loss, accuracy and model history.

```python
class History:
    def __init__(self):
       
        self.loss = []       # stores model loss
        self.accuracy = []   # stores model accuracy
        self.model = []      # stores model

    def log_server(self, model, client, loss_func):
        # Logging loss to history
        curr_loss = [(float)(get_dataset_loss(model, dataset, loss_func).detach()) 
											for dataset in client[0]]
        self.loss.append(sum(curr_loss)/len(curr_loss))

        # Logging accuracy to history
        curr_acc = [get_dataset_accuracy(model, dataset) for dataset in client[1]]
        self.accuracy.append(sum(curr_acc)/len(curr_acc))

        # Logging model to history
        self.model.append(model.state_dict())

    def log_client(self, model, dataset, loss):
        # Logging loss to history
        self.loss.append(loss)

        # Logging accuracy to history
        curr_acc = get_dataset_accuracy(model, dataset)
        self.accuracy.append(curr_acc)

        # Logging model to history
        self.model.append(model.state_dict())
```

**Functions to compute loss and accuracy**

```python
def loss_classifier(predictions, labels):

    loss = nn.CrossEntropyLoss(reduction="mean")
    return loss(predictions, labels)

def get_dataset_loss(model, dataset, loss_func):
    # Compute loss of a model on given dataset

    total_loss = 0
    for batch in dataset:
        features, labels = batch
        predictions = model(features)
        total_loss += loss_func(predictions, labels)

    avg_loss = total_loss/len(dataset)
    return avg_loss

def get_accuracy(predictions, labels):

     _, predicted = torch.max(predictions, dim=1)
     accuracy = torch.sum(predicted == labels).item()/len(predicted)
     return accuracy

def get_dataset_accuracy(model, dataset):

    total_accuracy = 0
    for batch in dataset:
        features, labels = batch
        predictions = model(features)
        curr_accuracy = get_accuracy(predictions, labels)
        total_accuracy += curr_accuracy
    
    avg_accuracy = (total_accuracy/len(dataset))*100
    return avg_accuracy

def train(model, batch, loss_func):
    images, labels = batch
    preds = model(images)
    loss = loss_func(preds, labels)
    return loss

def diff_squared_sum(model1, model2):
    w1 = model1.linear.weight
    w2 = model2.linear.weight
    w = (w1-w2).pow(2).sum()

    b1 = model1.linear.bias
    b2 = model2.linear.bias
    b = (b1-b2).pow(2).sum()
    
    dss = w + b
   

    return dss
```

**Aggregating function for FedAvg and FedProx**

```python
def get_model_avg(history):

    # Setup
    model_state = history.model
    sd = model_state[0] # create state dictionary

    # Zero all the values of state dictionary
    for key in sd:
        sd[key] = 0

    # Average all parameters
    n = len(model_state)
    for state in model_state:
        for key in sd:
            sd[key] += state[key]*(1/n)
    
    
    # Recreate model and load averaged state_dict (or use modelA/B)
    model = LogisticRegression()
    model.load_state_dict(sd)

    return model
```

**Aggregating function for qFedAvg**

```python
def q_aggregate(server_model, deltas, hs):
    num_clients = len(deltas)

    de = np.sum(np.asarray(hs))
    
    # Scale client deltas by multiplying (1/denominator)
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([(layer * 1.0 / de) for layer in client_delta])

    # Sum scaled client deltas
    sum_delta = deltas[0]
    for i in range(len(scaled_deltas)):
        if(i > 0):
            sum_delta = [(sd+d) for sd, d in zip(sum_delta, scaled_deltas[i])]

 
    # Update server model
    model = deepcopy(server_model)
   

    with torch.no_grad():
        model.linear.weight -= sum_delta[0]
        model.linear.bias -=  sum_delta[1]

    return model
```

**Client Function for FedAvg and FedProx**

```python
def clientUpdate(k, client, server_model, history, learning_rate, epochs, 
									mu, loss_func, type, q):
    local_model = deepcopy(server_model)
    local_loss = 0
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    
    for i in range(epochs):
        epoch_loss = 0
        
        # Training
        for batch in client:
            loss = train(local_model, batch, loss_func)
            if(type=="prox"):
                loss += (mu/2)*(diff_squared_sum(local_model, server_model))

            epoch_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        

        epoch_loss = epoch_loss/len(client)
        local_loss = epoch_loss

    history.log_client(local_model, client, local_loss)
    return history
```

**Client Function for qFedAvg**

```python
def q_clientUpdate(k, client, server_model, history, learning_rate,
									 epochs, loss_func, type, q):

    # Communicate the latest model
    local_model = deepcopy(server_model)

    # Train server model
    local_loss = 0
    optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
    
    for i in range(epochs):
        epoch_loss = 0
        
        # Training
        for batch in client:
            loss = train(local_model, batch, loss_func)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss = epoch_loss/len(client)
        local_loss = epoch_loss

    history.log_client(local_model, client, local_loss)

     # Compute loss on the whole training data
    comp_loss = 0
    for batch in client:
        comp_loss += train(local_model, batch, loss_func)
    
    comp_loss = comp_loss/(len(client))

    # Get difference between the weights
    delta_ws = [(x - y)*(1/lr) for x, y in zip(server_model.state_dict().values(), local_model.state_dict().values())]

    # Calc deltas
    q_loss = comp_loss.detach()
    delta = [np.float_power(q_loss +1e-10, q) * delta_w for delta_w in delta_ws]
  
    # Calc h
    h =  ((q * np.float_power(q_loss+1e-10, (q-1)) * normal(delta_ws)) + ((1/learning_rate) * np.float_power(q_loss+1e-10, q)))
    
return history, delta, h
```

**Federated Algorithms**

I have implemented three federated algorithms.

1. FedAvg
2. FedProx
3. qFedAvg

**FedAvg**

```python
def FedAvg(server_model, clients, client_fraction=1, rounds=10, epochs=10, 
learning_rate=0.01, decay=1, straggler_percent=0, type="avg"):
    
		model = deepcopy(server_model)
    loss_func = loss_classifier
    K = len(clients[0])# number of clients

    server_history = History()
    server_history.log_server(model, clients, loss_func)

    for i in range(rounds):
        # Calculate number of clients with given fraction
        m = max(math.floor(K*client_fraction), 1)

        # Select random set on m clients
        s = random.sample(range(0, K), m) # for now selecting all clients

        local_history = History()
        server_model = deepcopy(model)

        num_straggler = straggler_percent*len(s)
        straggler_epochs = max(int(epochs*(0.2)),1)

        for j in range(len(s)):
            if(j <= math.floor(len(s)-int(num_straggler))):
                # Not a Straggler
                local_history = clientUpdate(s[j], clients[0][s[j]], server_model, 
																local_history,learning_rate, epochs, loss_func, 
																type=type)

        model = get_model_avg(local_history)
        server_history.log_server(model, clients, loss_func)

    hist = server_history
  
    return model, server_history
```

**FedProx**

```python
def FedProx(server_model, clients, client_fraction=1, rounds=10, epochs=10, 
learning_rate=0.01, decay=1, type="prox", straggler_percent=0, mu=0):

    model = deepcopy(server_model)
    loss_func = loss_classifier
    K = len(clients[0])# number of clients

    server_history = History()
    server_history.log_server(model, clients, loss_func)

    for i in range(rounds):
        # Calculate number of clients with given fraction
        m = max(math.floor(K*client_fraction), 1)

        # Select random set on m clients
        s = random.sample(range(0, K), m) # for now selecting all clients

        local_history = History()
        server_model = deepcopy(model)

        num_straggler = straggler_percent*len(s)
        straggler_epochs = max(int(epochs*(0.2)),1)

        for j in range(len(s)):

            if(j <= math.floor(len(s)-int(num_straggler))):
                # Not a Straggler
                local_history = clientUpdate(s[j], clients[0][s[j]], 
																server_model, local_history,learning_rate, 
																epochs, mu, loss_func, type=type)
            else:
                # Straggler
                local_history = clientUpdate(s[j], clients[0][s[j]], 
																server_model, local_history,learning_rate, 
																straggler_epochs, mu, loss_func, type=type)

        model = get_model_avg(local_history)
        server_history.log_server(model, clients, loss_func)

    hist = server_history 
    return model, server_history
```

**qFedAvg**

```python
def qFed(server_model, clients, client_fraction=1, rounds=10, epochs=10, 
learning_rate=0.01, decay=1, type="sdg", straggler_percent=0, q=0):
   
    model = deepcopy(server_model)
    loss_func = loss_classifier
    K = len(clients[0])        # number of clients

    server_history = History()
    server_history.log_server(model, clients, loss_func)

    for i in range(rounds):
       
        # Calculate number of clients with given fraction
        m = max(math.floor(K*client_fraction), 1)

        # Select random set on m clients
        s = random.sample(range(0, K), m) # for now selecting all clients
        
        local_history = History()
        s_model = deepcopy(model)

        num_straggler = straggler_percent*len(s)
        straggler_epochs = max(int(epochs*(0.2)),1)

        deltas = []
        hs = []
        for j in range(len(s)):
            if(j <= math.floor(len(s)-int(num_straggler))):
                local_history, delta, h = q_clientUpdate(s[j], clients[0][s[j]], 
																					s_model, local_history,learning_rate, 
																					epochs, loss_func, q=q)
            else:
                local_history, delta, h = q_clientUpdate(s[j], clients[0][s[j]],
																					s_model, local_history,learning_rate,
																					straggler_epochs, loss_func, q=q)

            deltas.append(delta)
            hs.append(h)

        model = q_aggregate(s_model, deltas, hs)
        server_history.log_server(model, clients, loss_func)

    hist = server_history

    
    return model, server_history
```

## Analyzing Training and Test Results

**Clients with IID Dataset**

We can observe that qFedAvg results in significant convergence improvements relative to FedAvg and FedProx in IID settings with 0% and 40% stragglers. In both the IID settings, FedAvg and FedProx perform similarly. With FedAvg we drop the stragglers, whereas with FedProx(mu=0) we consider stragglers.

![legend.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/legend.png)

![iid_loss_str_0_crp.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iid_loss_str_0_crp.png)

![iid_loss_str_40_crp.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iid_loss_str_40_crp.png)

![iid_acc_str_0_crp.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iid_acc_str_0_crp.png)

![iid_acc_str_40_crp.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iid_acc_str_40_crp.png)

**Client with Non IID Dataset**

Relative to FedAvg and qFedProx, we can observe that FedProx results in significant convergence improvements in heterogeneous networks. We simulate different levels of systems heterogeneity by forcing 0%, 40% and 80% devices to be the stragglers (dropped by FedAvg).

We can observe that qFedAvg struggles to perform in Non IID settings. Performance of qFedAvg highly depends upon the hyper-parameters q and learning-rate.

![niid_loss_0.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid_loss_0.png)

![niid_acc_0.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid_acc_0.png)

![niid_loss_40.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid_loss_40.png)

![niid_acc_40.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid_acc_40.png)

![niid_loss_80.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid_loss_80.png)

![niid_acc_80.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niid_acc_80.png)

If we compare FedAvg and FedProx(µ = 0), we can observe that considering variable amounts of work performed by devices can help in convergence. 

With FedAvg with 40% and 80% stragglers, we can observe that it cannot converge efficiently. Comparing FedProx (µ = 0) with FedProx (µ > 0), we can see the effect of the proximal term. 

FedProx with µ > 0 leads to more stable convergence and enables otherwise divergent methods to converge, both in the presence of systems heterogeneity (40% and 80% stragglers) and without systems heterogeneity (0% stragglers).

**Clients with IID Synthetic Datasets**

The performance of FedAvg and FedProx for synthetic-dataset with IID settings is like that of the MNIST dataset with IID settings. However, FedProx(µ=0) performs than the other two.

![iidsyn_loss_0.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iidsyn_loss_0.png)

![iidsyn_loss_40.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iidsyn_loss_40.png)

![iidsyn_acc_0.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iidsyn_acc_0.png)

![iidsyn_acc_40.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/iidsyn_acc_40.png)

**Clients with Synthetic Non IID datasets**

For Synthetic data with non IID settings, FedProx(µ>0) shows smooth convergence compared the FedAvg and FedProx(µ=0).

![niidsyn_loss_0.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niidsyn_loss_0.png)

![niidsyn_acc_0.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niidsyn_acc_0.png)

![niidsyn_loss_40.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niidsyn_loss_40.png)

![niidsyn_acc_40.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/niidsyn_acc_40.png)

**Conclusion**

qFedAvg performs very well in IID settings, outperforming FedAvg and FedProx. Training with qFedAvg requires rigorous tuning of hyper-parameters like q and learning-rate. FedProx allows for variable amounts of work to be performed locally across devices and relies on a proximal term to help stabilize the method. In every heterogeneous setting, we can observe that FedProx provides smooth convergence.

# Security in Federated Learning

![poisioning.png](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/poisioning.png)

Federated learning comprises multiple client-server interactions. In each round, the server transmits the current global model to a set of nodes. These nodes train the transmitted model locally and send it to the server. Thereafter, the server aggregates these local models and updates the global model.

# Threats in Federated Learning

### Poisoning

All clients in federated-learning have access to the training data, so the possibility of adding manipulated data weights to global machine learning model is high.

Model updates taken from large group of clients during training can contain such poisoned updates. Such updates can impact the perfomance of the global model

**Poisioning attack classification**

1. Data Poisioning: Generating dirty samples to train the global model.
2. Model Poisioning: Modify the updated model before sending it to the central server for aggregation.
3. Data Modification:  Adding a shade or pattern of another class to a targeted class or random label swap of the training dataset.

### Backdoor Attacks

A backdoor attack is a method of inserting a malicious task into an existing model while maintaining the accuracy of the actual task. Identifying backdoor attacks can be difficult and time consuming, as the accuracy of actual ML tasks may not be immediately affected.

### GANs

GANs stands for generative adversarial networks. GANs are increasingly popular in big data and also apply to FL based approaches. It can be used for launching poisioning and inference attacks. GANs can be used to get training data through inference and use GANs to poison the training data.

### Malicious Server

Central servers play an important role in federated learning. Model and model parameters selection, aggregation of client updates and deployment of global model are performed by the central server. Therefore a compromised central server is a huge threat to clients privacy. Such servers can extract clients private data or manipulate the global model.

## Some Defence Techniques

### Sniper

Sniper is a filtering mechanism which is conducted by the global server to remove attackers from the global model. In this method we configure euclidean distance checks between the local models. 

Sniper can recognize honest users and drop attack success rate significantly even when multiple attackers are in the federated learning system

### Anomaly Detection

Anomaly detection can detect various attacks such as data poisoning, model poisoning or torjar threats. This technique mostly utilizes statistical and analytical methods in order to identify unexpected pattern or activity. Profile of the normal behavior helps the anomaly detection system to detect deviations.

### Data Sanitization

This technique utilizes an anomaly detector to filter out suspicious training data points. Data sanitization technique is a commonly used defence against data poisioning attac

# Proposal: Filtering mechanism to limit the impact of attackers on the global model

### Aim

To filter out statistical outliers and limit their impact

### Idea

$w_m = (1/K)\displaystyle\sum_{k=1}^K w_t^k$

$d_k = \sqrt {\sum _{i=1}^{n}  \left( w_{m}-w_{t}^k\right)^2 }$

$d_m = (1/K)\displaystyle\sum_{k=1}^K d_k$

For all  $d_k < d_m$

$w_{a} = (1/K)\displaystyle\sum_{k=1}^K w_t^k$

For all  $d_k > d_m$

$w_{b} = \displaystyle\sum_{k=1}^K w_t^k/(K*d_k)$

$w_{t+1} = w_a + w_b$

![proposal.svg](Report%20Federated%20Learning%206b8eead82f84416594d49deeb154a2c3/proposal.svg)

### Working

The server executes the **FedAvg** algorithm with the above filtering mechanism for several rounds of training.

1. First, the weights are initialized
2. It samples a fraction of ***C*** of the ***K*** clients.
3. The server sends the current round of weights to each client ***k.***
4. The weights for round ***t*** is denoted by ***wt***.
5. The client runs stochastic gradient descent (SDG) on their local data for ***E epochs*.**
6. The updated weights are sent back to the server.
7. Once the server receives the updated weights, it does the following:-
    1. Calculate the mean $w_m$ of all client weights .
    2. Stores the euclidean distance $d_k$ of client weights from $w_m$
    3. Calculates average distance $d_m$
    4. Take mean $w_a$ for all client weights  having distance between $w_k^t$ and $w_m$ is less the $d_m$ 
    5. Take mean $w_b$ and divide by $d_k$ for all clients having distance $d_k > d_m$