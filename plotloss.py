import pickle
import matplotlib.pylab as plt

losses = pickle.load(open('epoch-losses.p','rb'))
plt.plot(losses['loss'],'r'); plt.title('Training/Validation Loss')
plt.plot(losses['val_loss'],'b'); plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
