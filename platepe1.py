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
h = 0.001
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
# Optimizer     
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

# Boundary points
# Nb = 400
# xb = torch.rand(Nb,1, device=device, requires_grad=True)
# yb = torch.rand(Nb,1, device=device, requires_grad=True) 

# # Boundary conditions
# #left edge
# xbL = torch.zeros(Nb,1, device=device, requires_grad=True)
# ybL = torch.rand(Nb,1, device=device, requires_grad=True)
# #right edge
# xbR = torch.ones(Nb,1, device=device, requires_grad=True)
# ybR = torch.rand(Nb,1, device=device, requires_grad=True)
# #bottom edge
# xbB = torch.rand(Nb,1, device=device, requires_grad=True)
# ybB = torch.zeros(Nb,1, device=device, requires_grad=True)
# #top edge
# xbT = torch.rand(Nb,1, device=device, requires_grad=True)
# ybT = torch.ones(Nb,1, device=device, requires_grad=True)


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
    ENERGY_LOSS = U-W
  
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
    ENERGY_LOSS = U-W
  
    R_LOSS = ENERGY_LOSS
    
    # resd = (D/q)*(w_xxxx +2*w_xxyy+w_yyyy)-1
    
    # pde_loss = 
    # resd_loss = torch.mean(resd**2)
    
    # Boundary loss
    
    # w_bL = model(xbL,ybL)
    # w_xbL = torch.autograd.grad(w_bL,xbL,torch.ones_like(w_bL), create_graph=True)[0]
    # w_xxbL = torch.autograd.grad(w_xbL,xbL,torch.ones_like(w_xbL), create_graph=True)[0]
    # w_ybL = torch.autograd.grad(w_bL,ybL,torch.ones_like(w_bL), create_graph=True)[0]
    # w_yybL = torch.autograd.grad(w_ybL,ybL,torch.ones_like(w_ybL), create_graph=True)[0]
    # MxL = -(w_xxbL )
    
    # w_bR = model(xbR,ybR)
    # w_xbR = torch.autograd.grad(w_bR,xbR,torch.ones_like(w_bR), create_graph=True)[0]           
    # w_xxbR = torch.autograd.grad(w_xbR,xbR,torch.ones_like(w_xbR), create_graph=True)[0]
    # w_ybR = torch.autograd.grad(w_bR,ybR,torch.ones_like(w_bR), create_graph=True)[0]
    # w_yybR = torch.autograd.grad(w_ybR,ybR,torch.ones_like(w_ybR), create_graph=True)[0]
    # MxR = -(w_xxbR )
   
    # w_bB = model(xbB,ybB)
    # w_ybB = torch.autograd.grad(w_bB,ybB,torch.ones_like(w_bB), create_graph=True)[0]
    # w_yybB = torch.autograd.grad(w_ybB,ybB,torch.ones_like(w_ybB), create_graph=True)[0]
    # w_xbB = torch.autograd.grad(w_bB,xbB,torch.ones_like(w_bB), create_graph=True)[0]
    # w_xxbB = torch.autograd.grad(w_xbB,xbB,torch.ones_like(w_xbB), create_graph=True)[0]
    # MyB = -(w_yybB )

    # w_bT = model(xbT,ybT)
    # w_ybT = torch.autograd.grad(w_bT,ybT,torch.ones_like(w_bT), create_graph=True)[0]
    # w_yybT = torch.autograd.grad(w_ybT,ybT,torch.ones_like(w_ybT), create_graph=True)[0]
    # w_xbT = torch.autograd.grad(w_bT,xbT,torch.ones_like(w_bT), create_graph=True)[0]
    # w_xxbT = torch.autograd.grad(w_xbT,xbT,torch.ones_like(w_xbT), create_graph=True)[0]
    # MxT = -(w_xxbT )

    # BC_LOSS = torch.mean(w_bL**2)+torch.mean(w_bR**2)+torch.mean(w_bB**2)+torch.mean(w_bT**2) +torch.mean(MxL**2)+torch.mean(MyB**2)+torch.mean(MxR**2)+torch.mean(MxT**2)

    # total_loss = R_LOSS + 100000*BC_LOSS
    
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

w_pred = model(x_test,y_test).detach().cpu().numpy()
max_def = np.max(w_pred)
print(max_def)

# -----------------------------
# Plot
# -----------------------------
plt.figure()
plt.tricontourf(
    x_test.cpu().numpy().flatten(),
    y_test.cpu().numpy().flatten(),
    w_pred.flatten(),
    50
)
plt.colorbar()
plt.title("Plate Deflection (PINN)")
plt.show()

