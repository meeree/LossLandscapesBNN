# Generate all figures in paper by analyzing data/etc.

PATH_TO_DATA = 'C:/Users/jhazelde/BNBP-dev/src/pytorch_impl/HH_2000_model_200_I0_1.5_0.100000.pt'

model = BNN()
model.load_state_dict(torch.load(PATH_TO_DATA))

plt.imshow(model.Ws[1].weight.data.numpy(), cmap='seismic', aspect='auto')
plt.show()

plt.figure(figsize=(30,30))
W1 = model.Ws[0].weight.data.numpy()
vmin, vmax = W1.min(), W1.max()
print(vmin, vmax)
W1 = W1.reshape(10, 10, 28, 28)

for i in range(10):
    for j in range(10):     
        plt.subplot(10, 10, i + j * 10 + 1)
        plt.imshow(W1[i, j, :, :], cmap='seismic', interpolation='bilinear')
        plt.box(False)
        plt.xticks([])
        plt.yticks([])

plt.show()

plt.imshow(model.Ws[1].weight.data.numpy(), aspect = 'auto', cmap='seismic', interpolation='none')
plt.colorbar()
plt.show()
