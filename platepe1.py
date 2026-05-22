import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# Reproducebility
torch.manual_seed(0)
np.random.seed(0)
# Devise selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plate parameters
q = 10000
E= 200e9
h = 0.01
v = 0.3
D = (E*h**3)/(12*(1-v**2))


# Neural Network
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,64),
            
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),      
            nn.Tanh(),
            nn.Linear(64,64),      
            nn.Tanh(),
            nn.Linear(64,1)  
        )
    
    def forward(self,x,y):
        xy = torch.cat([x,y], dim=1)
        nn_out = self.net(xy)   
        bc_cond = (x*(1-x)*y*(1-y)).to(device)  # Enforce zero deflection at boundaries
        return nn_out * bc_cond 
        

model = PINN().to(device) 
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001) 
    
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=190,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=50,
    line_search_fn="strong_wolfe"
)
# Collocation points
N = 2000
x = torch.rand(N,1, requires_grad=True, device=device)
y = torch.rand(N,1, requires_grad=True, device=device)



# epochs = 5000

# for epoch  in range(epochs):

epochs_adam = 2750

for epoch  in range(epochs_adam):
    optimizer_adam.zero_grad()
    
    # PDE residual
    w = model(x,y)
    
    w_x = torch.autograd.grad(w,x,torch.ones_like(w),create_graph=True)[0]
    
    w_y = torch.autograd.grad(w,y,torch.ones_like(w),create_graph=True)[0]
    
    w_xx = torch.autograd.grad(w_x,x,torch.ones_like(w_x),create_graph=True)[0]
    
    w_yy = torch.autograd.grad(w_y,y,torch.ones_like(w_y),  create_graph=True)[0]
    
    w_xy = torch.autograd.grad(w_x,y,torch.ones_like(w_x),create_graph=True)[0]
    
    energy_density = (w_xx**2+ w_yy**2+ 2*v*w_xx*w_yy + (2*(1-v)*w_xy**2))
    U = 0.5*D*torch.mean(energy_density)*1.0
    W = torch.mean(q*w)
    ENERGY_LOSS = (U-W)
  
    R_LOSS = ENERGY_LOSS
    
    total_loss = R_LOSS

    total_loss.backward()
    optimizer_adam.step()

def closure():
    
    optimizer_lbfgs.zero_grad()
    
    # PDE residual
    w = model(x,y)
    
    w_x = torch.autograd.grad(w,x,torch.ones_like(w),create_graph=True)[0]
    
    w_y = torch.autograd.grad(w,y,torch.ones_like(w),create_graph=True)[0]
    
    w_xx = torch.autograd.grad(w_x,x,torch.ones_like(w_x),create_graph=True)[0]
    
    w_yy = torch.autograd.grad(w_y,y,torch.ones_like(w_y),  create_graph=True)[0]
    
    w_xy = torch.autograd.grad(w_x,y,torch.ones_like(w_x),create_graph=True)[0]
    
    
    
    energy_density = (w_xx**2+ w_yy**2+ 2*v*w_xx*w_yy + (2*(1-v)*w_xy**2))
    U = 0.5*D*torch.mean(energy_density)*1.0
    W = torch.mean(q*w)
    ENERGY_LOSS = (U-W)
  
    R_LOSS = ENERGY_LOSS
    
    
    total_loss = R_LOSS

    total_loss.backward()
    print("Loss:", total_loss.item())
            # if epoch % 500 == 0:
            #     print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}, PDE Loss: {resd_loss.item():.6f}, BC Loss: {BC_LOSS.item():.6f}")
    return total_loss 

optimizer_lbfgs.step(closure)


x_test = torch.linspace(0,1,50)
y_test = torch.linspace(0,1,50)

X,Y = torch.meshgrid(x_test,y_test)

x_test = X.reshape(-1,1).to(device)
y_test = Y.reshape(-1,1).to(device)
x_test.requires_grad_(True)
y_test.requires_grad_(True)
w_pred = model(x_test,y_test)

w_x = torch.autograd.grad(w_pred,x_test,torch.ones_like(w_pred),create_graph=True)[0]
w_y = torch.autograd.grad(w_pred,y_test,torch.ones_like(w_pred),create_graph=True)[0]
w_xx = torch.autograd.grad(w_x,x_test,torch.ones_like(w_x),create_graph=True)[0]
w_yy = torch.autograd.grad(w_y,y_test,torch.ones_like(w_y),  create_graph=True)[0]
w_xy = torch.autograd.grad(w_x,y_test,torch.ones_like(w_x),create_graph=True)[0]

z = h/2
sigma_x = -(E*z/(1-v**2))*(w_xx + v*w_yy)
sigma_y = -(E*z/(1-v**2))*(w_yy + v*w_xx)
tau_xy = -(E*z/(1-v))*w_xy

w_np = w_pred.cpu().detach().numpy()
sigma_x_np = sigma_x.cpu().detach().numpy()
sigma_y_np = sigma_y.cpu().detach().numpy()
tau_xy_np = tau_xy.cpu().detach().numpy()

print("Max deflection:", np.max(w_np))

print("Max sigma_x:", np.max(sigma_x_np))

print("Max sigma_y:", np.max(sigma_y_np))

print("Max tau_xy:", np.max(tau_xy_np))



# # -----------------------------
# # Plot
# # -----------------------------


def plot_countour(x,title):
    plt.figure()
    plt.tricontourf(
        x_test.detach().cpu().numpy().flatten(),
        y_test.detach().cpu().numpy().flatten(),
        x.flatten(),
        50
    )
    plt.colorbar()
    plt.title(title)
    plt.show()
   

plot_countour(w_np,"Plate Deflection (PINN)")
plot_countour(sigma_x_np,"Plate Stress_xx (PINN)")
plot_countour(sigma_y_np,"Plate Stress_yy (PINN)")
plot_countour(tau_xy_np,"Plate Shear Stress_xy (PINN)")

    
